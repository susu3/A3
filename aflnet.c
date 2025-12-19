#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>

#include "alloc-inl.h"
#include "aflnet.h"

// Mapping from original state IDs to compact IDs starting from 1

static u32 message_code_counter = 0;
khash_t(32) *message_code_map = NULL;

void init_message_code_map(){
  message_code_map = kh_init(32);
}

void destroy_message_code_map(){
  kh_destroy(32, message_code_map);
}

u32 get_mapped_message_code (u32 ori_message_code){
  u32 mapped_message_code = 0;
  khiter_t k = kh_get(32, message_code_map, ori_message_code);
  if (k == kh_end(message_code_map)) {
    int ret;
    k = kh_put(32, message_code_map, ori_message_code, &ret);
    message_code_counter++;
    kh_value(message_code_map, k) = message_code_counter;

    mapped_message_code = message_code_counter;
  }
  else {
    mapped_message_code = kh_value(message_code_map, k);
  }

  return mapped_message_code;
}

// Protocol-specific functions for extracting requests and responses


region_t* extract_requests_mqtt(unsigned char* buf, unsigned int buf_size, unsigned int* region_count_ref)
{
  char *mem;
  unsigned int mem_count = 0;
  unsigned int mem_size = 1024;
  unsigned int region_count = 0;
  unsigned int cur_start = 0;
  unsigned int cur_end = 0;
  region_t *regions = NULL;
  mem=(char *)ck_alloc(mem_size);
  while(cur_start < buf_size)
  {
		if ((buf_size - cur_start) == 1) {
			region_count++;
			regions = (region_t *)ck_realloc(regions, region_count * sizeof(region_t));
			regions[region_count - 1].start_byte = cur_start;
			regions[region_count - 1].end_byte = buf_size - 1;
			regions[region_count - 1].state_sequence = NULL;
			regions[region_count - 1].state_count = 0;	
			break;	
		}
    // Read the packet header
    memcpy(&mem[mem_count], buf + cur_start, 2);
    cur_start = cur_start + 2;
    // Check the packet length and update current_end
    // mem[0] is Message Type. mem[1] is Msg Len.
    if(mem[1] >= 0) 
      cur_end = cur_start + mem[1] - 1;
    else
      cur_end = buf_size;
    // Create a region for every request
		region_count++;
		regions = (region_t *)ck_realloc(regions, region_count * sizeof(region_t));
		regions[region_count - 1].start_byte = cur_start - 2;
		regions[region_count - 1].end_byte = cur_end;
		regions[region_count - 1].state_sequence = NULL;
		regions[region_count - 1].state_count = 0;
    // Update the indices
    mem_count = 0;
    cur_start = cur_end + 1;
    cur_end = cur_start;
  }
  if(mem) ck_free(mem);
  //in case region_count equals zero, it means that the structure of the buffer is broken
  //hence we create one region for the whole buffer
	if ((region_count == 0) && (buf_size > 0)) {
		regions = (region_t *)ck_realloc(regions, sizeof(region_t));
		regions[0].start_byte = 0;
		regions[0].end_byte = buf_size - 1;
		regions[0].state_sequence = NULL;
		regions[0].state_count = 0;
		region_count = 1;
	}
	*region_count_ref = region_count;
	return regions;
}

// Convert a 2-byte big-endian value to a 2-byte little-endian value or vice versa
#define ushort_be_to_se(v) ((v & 0xff) << 8) + ((v & 0xff00) >> 8)

#pragma pack (push, 1) //Ensure compact arrangement without padding bytes
typedef struct mbap_be{
  unsigned short tid; //transaction id: 2 Bytes
  unsigned short protocol; //protocol identifier: 2 Bytes
  unsigned short length; //length: 2 Bytes = uid + data fields
  unsigned char uid; //unit id: 1 Bytes
  unsigned char fid; //function code: 1 Bytes
}mbap_be;
#pragma pack (pop)

//buf: seed block; region_count_ref: region count
//将buf分割为多个field，并指出每个field的大小，返回
//按照每个request来划分
region_t* extract_requests_modbus(unsigned char* buf, unsigned int buf_size, unsigned int* region_count_ref)
{

    unsigned char* end_ptr = buf + buf_size -1;
    unsigned char* cur_ptr = buf;
    unsigned int region_count = 0;
    unsigned int cur = 0;
    region_t* regions = NULL;

    if (buf == NULL || buf_size == 0) {
        *region_count_ref = region_count;
        return regions;
    }

    while (cur_ptr <= end_ptr) {

        unsigned int remaining_buf_size = end_ptr - cur_ptr + 1;
        //MBAP + function code = 8 Bytes
        if (remaining_buf_size >= sizeof(mbap_be)) { //remaining_buf_size >=8
            region_count ++;
            regions = (region_t *)ck_realloc(regions, region_count*sizeof(region_t)); //Re-allocate a buffer, checking for issues and zeroing any newly-added tail

            regions[region_count-1].state_sequence = NULL;
            regions[region_count-1].state_count = 0;
            regions[region_count-1].start_byte = cur;
            // check data region
            mbap_be *header = (mbap_be *)cur_ptr;
            // data field <= 252 bytes for valid packet
            // length field = uid + fid + data <= 254
            unsigned short remaining_packet_length = ushort_be_to_se(header->length);
            remaining_packet_length = (remaining_packet_length > 254) ? 254 : remaining_packet_length;
            remaining_packet_length = (remaining_buf_size - 6 >= remaining_packet_length) ? remaining_packet_length : remaining_buf_size - 6;

            if (remaining_packet_length >= 0) {
                cur = cur + sizeof(mbap_be) - 2 + remaining_packet_length - 1;
                regions[region_count -1].end_byte = cur++;
                cur_ptr = cur_ptr + sizeof(mbap_be) - 2 + remaining_packet_length;
            }else {
                break;
            }

        }else {
            //malformed
            if (remaining_buf_size > 0) {
                region_count = region_count + 1;
                regions = (region_t *)ck_realloc(regions, region_count*sizeof(region_t));
                regions[region_count-1].start_byte = cur;
                regions[region_count-1].end_byte = cur + remaining_buf_size -1;
                regions[region_count-1].state_sequence = NULL;
                regions[region_count-1].state_count = 0;
            }
            break;
        }
    }

    *region_count_ref = region_count; //region_count表示buf中一共有多少个field
    return regions; //将buf中每个field的开始结束位置标记
}

region_t *extract_requests_iec104(unsigned char *buf, unsigned int buf_size, unsigned int *region_count_ref) {

    unsigned char *end_ptr = buf + buf_size;
    unsigned char *cur_ptr = buf;
    unsigned int cur = 0;
    unsigned int region_count = 0;
    unsigned int remaining_buf = 0;
    unsigned short length = 0;

    region_t *regions = NULL;
    unsigned char start_byte = 0x68;

    if (buf == NULL || buf_size == 0) {
        *region_count_ref = region_count;
        return regions;
    }

    while (cur_ptr < end_ptr) {
        remaining_buf = end_ptr - cur_ptr;
        // Check if the first bytes are 0x68
        if (*cur_ptr == start_byte && (cur_ptr + 1) != end_ptr) {
            length = *(cur_ptr + 1);
            if (length + 2 <= remaining_buf) {    //china standard length: length <= 253
                region_count++;
                regions = (region_t *)ck_realloc(regions, region_count * sizeof(region_t));
                regions[region_count - 1].start_byte = cur;
                regions[region_count - 1].end_byte = cur + length + 1;
                regions[region_count - 1].state_sequence = NULL;
                regions[region_count - 1].state_count = 0;

                cur = cur + length + 2;
                cur_ptr = cur_ptr + length + 2;
            } else {
                region_count++;
                regions = (region_t *)ck_realloc(regions, region_count * sizeof(region_t));
                regions[region_count - 1].start_byte = cur;
                regions[region_count - 1].end_byte = cur + remaining_buf - 1;
                regions[region_count - 1].state_sequence = NULL;
                regions[region_count - 1].state_count = 0;

                break;
            }
        } else {
            cur_ptr++;
            cur++;
        }
    }
    // in case region_count equals zero, it means that the structure of the buffer
    // is broken hence we create one region for the whole buffer
    if ((region_count == 0) && (buf_size > 0)) {
        regions = (region_t *)ck_realloc(regions, sizeof(region_t));
        regions[0].start_byte = 0;
        regions[0].end_byte = buf_size - 1;
        regions[0].state_sequence = NULL;
        regions[0].state_count = 0;
        region_count = 1;
    }

    *region_count_ref = region_count;
    return regions;
}

region_t *extract_requests_ethernetip(unsigned char *buf, unsigned int buf_size, unsigned int *region_count_ref) {
    region_t *regions = NULL;
    unsigned int region_count = 0;
    unsigned int cur = 0;
    unsigned char *cur_ptr = buf;
    unsigned char *end_ptr = buf + buf_size;
    unsigned int remaining_buf;
    unsigned int length;

    // Check if the buffer is empty or too small
    if (buf_size == 0 || buf == NULL) {
        *region_count_ref = region_count;
        return regions;
    }

    while (cur_ptr < end_ptr) {
        remaining_buf = end_ptr - cur_ptr;
        
        // Check for Ethernet/IP encapsulation header (24 bytes)
        if (remaining_buf >= 24) {
            region_count++;
            regions = (region_t *)ck_realloc(regions, region_count * sizeof(region_t));
            regions[region_count - 1].start_byte = cur;
            regions[region_count - 1].state_sequence = NULL;
            regions[region_count - 1].state_count = 0;

            cur = cur + 24; //point to the beginning of the data field

            // Check for remaining data after header
            // Get length from bytes 3-4 (little-endian)
            length = (cur_ptr[3] << 8) | cur_ptr[2];

            if (remaining_buf - 24 >= length) {
                regions[region_count - 1].end_byte = cur + length - 1;
                cur = cur + length;
                cur_ptr = cur_ptr + 24 + length;
            }else{
                regions[region_count - 1].end_byte = cur + remaining_buf - 25;
                cur = cur + remaining_buf - 24;
                cur_ptr = end_ptr;
            }
        } else {
            // Buffer too small, create one region for whole buffer
            region_count++;
            regions = (region_t *)ck_realloc(regions, region_count * sizeof(region_t));
            regions[region_count - 1].start_byte = cur;
            regions[region_count - 1].end_byte = cur + remaining_buf - 1;
            regions[region_count - 1].state_sequence = NULL;
            regions[region_count - 1].state_count = 0;
            break;
        }
    }

    // In case region_count equals zero, create one region for the whole buffer
    if ((region_count == 0) && (buf_size > 0)) {
        regions = (region_t *)ck_realloc(regions, sizeof(region_t));
        regions[0].start_byte = 0;
        regions[0].end_byte = buf_size - 1;
        regions[0].state_sequence = NULL;
        regions[0].state_count = 0;
        region_count = 1;
    }

    *region_count_ref = region_count;
    return regions;
}

//SLMP ASCII code
region_t *extract_requests_slmpa(unsigned char* buf, unsigned int buf_size, unsigned int* region_count_ref){
  unsigned char* end_ptr = buf + buf_size;
  unsigned char* cur_ptr = buf;
  unsigned int cur = 0;
  unsigned int region_count = 0;
  unsigned int remaining_buf = 0;
  unsigned long length = 0;

  region_t* regions = NULL;
  if (buf == NULL || buf_size == 0) {
    *region_count_ref = region_count;
    return regions;
  }

  unsigned char start_byte[] = {0x35, 0x34, 0x30, 0x30};
  unsigned char start_byte_end[] = {0x30, 0x30, 0x30, 0x30};
  unsigned char start_byteN[] = {0x35, 0x30, 0x30, 0x30};
  unsigned short subheader_length = 0;

  while (cur_ptr <= end_ptr - 12) { 

    remaining_buf = end_ptr - cur_ptr;

    if(memcmp(cur_ptr, start_byte, 4) == 0){
      if(memcmp(cur_ptr+8, start_byte_end, 4) == 0){
        subheader_length = 12;
      }else{
        cur_ptr++;
        cur++;
        continue;
      }
    }else if(memcmp(cur_ptr, start_byteN, 4) == 0){
      subheader_length = 4;
    }else{
      cur_ptr++;
      cur++;
      continue;
    }

    // Check if enough bytes for minimum SLMP ASCII header 
    if (remaining_buf >= 14 + subheader_length) {
      region_count++;
      regions = (region_t *)ck_realloc(regions, region_count * sizeof(region_t));
      regions[region_count - 1].start_byte = cur;
      regions[region_count - 1].state_sequence = NULL;
      regions[region_count - 1].state_count = 0;

      cur = cur + subheader_length + 10; //point to beginning of length field
      cur_ptr = cur_ptr + subheader_length + 10;

      // Get length from ASCII hex chars
      char len_str[5];
      memcpy(len_str, cur_ptr, 4);
      len_str[4] = '\0';
      length = strtol(len_str, NULL, 16);

      if (remaining_buf >= subheader_length + 14 + length) {
        regions[region_count - 1].end_byte = cur + 4 + length - 1;
        cur = cur + 4 + length;
        cur_ptr = cur_ptr + 4 + length;
      } else {
        regions[region_count - 1].end_byte = buf_size - 1;
        break;
      }
    } else {
      // Buffer too small, create one region for whole buffer
      region_count++;
      regions = (region_t *)ck_realloc(regions, region_count * sizeof(region_t));
      regions[region_count - 1].start_byte = cur;
      regions[region_count - 1].end_byte = cur + remaining_buf - 1;
      regions[region_count - 1].state_sequence = NULL;
      regions[region_count - 1].state_count = 0;
      break;
    }
  }

  // In case region_count equals zero, create one region for the whole buffer
  if ((region_count == 0) && (buf_size > 0)) {
    regions = (region_t *)ck_realloc(regions, sizeof(region_t));
    regions[0].start_byte = 0;
    regions[0].end_byte = buf_size - 1;
    regions[0].state_sequence = NULL;
    regions[0].state_count = 0;
    region_count = 1;
  }

  *region_count_ref = region_count;
  return regions;
}

//SLMP Binary code
region_t *extract_requests_slmpb(unsigned char* buf, unsigned int buf_size, unsigned int* region_count_ref){
  unsigned char* end_ptr = buf + buf_size;
  unsigned char* cur_ptr = buf;
  unsigned int cur = 0;
  unsigned int region_count = 0;
  unsigned int remaining_buf = 0;
  unsigned long length = 0;

  region_t* regions = NULL;
  if (buf == NULL || buf_size == 0) {
    *region_count_ref = region_count;
    return regions;
  }

  unsigned char start_byte[] = {0x54, 0x00};
  unsigned char start_byte_end[] = {0x00, 0x00};
  unsigned char start_byteN[] = {0x50, 0x00};
  unsigned short subheader_length = 0;

  while (cur_ptr <= end_ptr - 6) { 

    remaining_buf = end_ptr - cur_ptr;

    if(memcmp(cur_ptr, start_byte, 2) == 0){
      if(memcmp(cur_ptr+4, start_byte_end, 2) == 0){
        subheader_length = 6;
      }else{
        cur_ptr++;
        cur++;
        continue;
      }
    }else if(memcmp(cur_ptr, start_byteN, 2) == 0){
      subheader_length = 2;
    }else{
      cur_ptr++;
      cur++;
      continue;
    }

    // Check if enough bytes for minimum SLMP binary header 
    if (remaining_buf >= 7 + subheader_length) {
      region_count++;
      regions = (region_t *)ck_realloc(regions, region_count * sizeof(region_t));
      regions[region_count - 1].start_byte = cur;
      regions[region_count - 1].state_sequence = NULL;
      regions[region_count - 1].state_count = 0;

      cur = cur + subheader_length + 5; //point to beginning of length field
      cur_ptr = cur_ptr + subheader_length + 5;
      
      length = length = (cur_ptr[1] << 8) | cur_ptr[0];

      if (remaining_buf >= subheader_length + 7 + length) {
        regions[region_count - 1].end_byte = cur + 2 + length - 1;
        cur = cur + 2 + length;
        cur_ptr = cur_ptr + 2 + length;
      } else {
        regions[region_count - 1].end_byte = buf_size - 1;
        break;
      }
    } else {
      // Buffer too small, create one region for whole buffer
      region_count++;
      regions = (region_t *)ck_realloc(regions, region_count * sizeof(region_t));
      regions[region_count - 1].start_byte = cur;
      regions[region_count - 1].end_byte = cur + remaining_buf - 1;
      regions[region_count - 1].state_sequence = NULL;
      regions[region_count - 1].state_count = 0;
      break;
    }
  }

  // In case region_count equals zero, create one region for the whole buffer
  if ((region_count == 0) && (buf_size > 0)) {
    regions = (region_t *)ck_realloc(regions, sizeof(region_t));
    regions[0].start_byte = 0;
    regions[0].end_byte = buf_size - 1;
    regions[0].state_sequence = NULL;
    regions[0].state_count = 0;
    region_count = 1;
  }

  *region_count_ref = region_count;
  return regions;

}

//OPC UA Connection Protocol
region_t *extract_requests_opcuacp(unsigned char* buf, unsigned int buf_size, unsigned int* region_count_ref){
  unsigned char* end_ptr = buf + buf_size;
  unsigned char* cur_ptr = buf;
  unsigned int cur = 0;
  unsigned int region_count = 0;
  unsigned int remaining_buf = 0;
  unsigned long length = 0;

  const char* messagetype[] = {
    "MSG",
    "OPN", 
    "CLO",
    "HEL",
    "ERR",
    "ACK",
    "RHE"
  };

  region_t* regions = NULL;
  if (buf == NULL || buf_size == 0) {
    *region_count_ref = region_count;
    return regions;
  }

  // Buffer to hold 3 bytes plus null terminator
  char msg_type[4] = {0};
  int flag = 0; //0: not found; 1: found

  while (cur_ptr <= end_ptr - 8) { 
    remaining_buf = end_ptr - cur_ptr;
    flag = 0;

    memcpy(msg_type, cur_ptr, 3);
    msg_type[3] = '\0';

    for (int i = 0; i < sizeof(messagetype)/sizeof(messagetype[0]); i++) {
      if (strncmp(msg_type, messagetype[i], 3) == 0) {
        flag = 1;
        break;
      }
    }

    if(flag == 0){
      cur_ptr++;
      cur++;
      continue;
    }

    region_count++;
    regions = (region_t *)ck_realloc(regions, region_count * sizeof(region_t));
    regions[region_count - 1].start_byte = cur;
    regions[region_count - 1].state_sequence = NULL;
    regions[region_count - 1].state_count = 0;
        
    // Extract length from bytes 4-7 (little-endian)
    length = (cur_ptr[7] << 24) | (cur_ptr[6] << 16) | (cur_ptr[5] << 8) | cur_ptr[4];

    if (remaining_buf >= length) {
      regions[region_count - 1].end_byte = cur + length - 1;
      cur = cur + length;
      cur_ptr = cur_ptr + length;
    } else {
      regions[region_count - 1].end_byte = buf_size - 1;
      break;
    }
  }

  // In case region_count equals zero, create one region for the whole buffer
  if ((region_count == 0) && (buf_size > 0)) {
    regions = (region_t *)ck_realloc(regions, sizeof(region_t));
    regions[0].start_byte = 0;
    regions[0].end_byte = buf_size - 1;
    regions[0].state_sequence = NULL;
    regions[0].state_count = 0;
    region_count = 1;
  }

  *region_count_ref = region_count;
  return regions;
}

region_t *extract_requests_dnp3(unsigned char* buf, unsigned int buf_size, unsigned int* region_count_ref){
  unsigned char *end_ptr = buf + buf_size;
  unsigned char *cur_ptr = buf;
  unsigned int cur = 0;
  unsigned int region_count = 0;
  unsigned int remaining_buf = 0;
  unsigned short length = 0;
  unsigned short data_field_number = 0;
  unsigned short remaining_data = 0;
  unsigned short message_length = 0;

  region_t *regions = NULL;
  unsigned char start_byte[] = {0x05, 0x64};

  if (buf == NULL || buf_size == 0) {
    *region_count_ref = region_count;
    return regions;
  }

  while (cur_ptr < end_ptr) { 

    remaining_buf = end_ptr - cur_ptr;

    if(remaining_buf < 2 || memcmp(cur_ptr, start_byte, 2) != 0){
      cur_ptr++;
      cur++;
      continue;
    }

    // Check if enough bytes for minimum DNP3 header 
    if (remaining_buf >= 8) {

      length = cur_ptr[2];   //length is in bytes 3-4
      if(length < 5){
        cur_ptr++;
        cur++;
        continue;
      }

      data_field_number = (length - 5)/16;
      remaining_data = length - 5 - data_field_number*16;
      message_length = 2 + length + 2 + data_field_number*2 + (remaining_data > 0 ? 2 : 0);

      region_count++;
      regions = (region_t *)ck_realloc(regions, region_count * sizeof(region_t));
      regions[region_count - 1].start_byte = cur;
      regions[region_count - 1].state_sequence = NULL;
      regions[region_count - 1].state_count = 0;
    
      if (remaining_buf >= message_length) {
        regions[region_count - 1].end_byte = cur + message_length - 1;
        cur = cur + message_length;
        cur_ptr = cur_ptr + message_length;
      } else {
        regions[region_count - 1].end_byte = buf_size - 1;
        break;
      }
    } else {
      // Buffer too small, create one region for whole buffer
      region_count++;
      regions = (region_t *)ck_realloc(regions, region_count * sizeof(region_t));
      regions[region_count - 1].start_byte = cur;
      regions[region_count - 1].end_byte = cur + remaining_buf - 1;
      regions[region_count - 1].state_sequence = NULL;
      regions[region_count - 1].state_count = 0;
      break;
    }
  }

  // In case region_count equals zero, create one region for the whole buffer
  if ((region_count == 0) && (buf_size > 0)) {
    regions = (region_t *)ck_realloc(regions, sizeof(region_t));
    regions[0].start_byte = 0;
    regions[0].end_byte = buf_size - 1;
    regions[0].state_sequence = NULL;
    regions[0].state_count = 0;
    region_count = 1;
  }

  *region_count_ref = region_count;
  return regions;
}

region_t *extract_requests_bacnetip(unsigned char* buf, unsigned int buf_size, unsigned int* region_count_ref){
  unsigned char *end_ptr = buf + buf_size;
  unsigned char *cur_ptr = buf;
  unsigned int cur = 0;
  unsigned int region_count = 0;
  unsigned int remaining_buf = 0;
  unsigned short length = 0;

  region_t *regions = NULL;
  unsigned char start_byte = 0x81;

  if (buf == NULL || buf_size == 0) {
    *region_count_ref = region_count;
    return regions;
  }

  while (cur_ptr < end_ptr) {
    remaining_buf = end_ptr - cur_ptr;
    
    // Check if the first bytes are 0x81
    if (*cur_ptr != start_byte) {
      cur_ptr++;
      cur++;
      continue;
    }

    region_count++;
    regions = (region_t *)ck_realloc(regions, region_count * sizeof(region_t));
    regions[region_count - 1].start_byte = cur;
    regions[region_count - 1].state_sequence = NULL;
    regions[region_count - 1].state_count = 0;
    
    if (remaining_buf >= 4) {
        // Add bounds checking to prevent excessive memory allocation
        // BACnet/IP length field should be reasonable (4 bytes to 1476 bytes typically)
        if (length < 4 || length > 65535 || length > remaining_buf) {
            // Invalid length field, skip this malformed packet and continue parsing
            cur_ptr++;
            cur++;
            region_count--; // Remove the invalid region we just added
            continue;
        }
      length = (cur_ptr[2] << 8) | cur_ptr[3];
      if(remaining_buf >= length){
        regions[region_count - 1].end_byte = cur + length -1;
        cur = cur + length;
        cur_ptr = cur_ptr + length;
      } else {
        regions[region_count - 1].end_byte = buf_size - 1;
        break;
      }
    } else {
      regions[region_count - 1].end_byte = buf_size - 1;
      break;
    }
  }

  // in case region_count equals zero, it means that the structure of the buffer
  // is broken hence we create one region for the whole buffer
  if ((region_count == 0) && (buf_size > 0)) {
    regions = (region_t *)ck_realloc(regions, sizeof(region_t));
    regions[0].start_byte = 0;
    regions[0].end_byte = buf_size - 1;
    regions[0].state_sequence = NULL;
    regions[0].state_count = 0;
    region_count = 1;
  }

    *region_count_ref = region_count;
    return regions;
}

//same as extract request
//buf: response data; state_count_ref: 
unsigned int* extract_response_codes_modbus(unsigned char* buf, unsigned int buf_size, unsigned int* state_count_ref)
{
  unsigned char* end_ptr = buf + buf_size -1;
  unsigned char* cur_ptr = buf;
  unsigned int* state_sequence = NULL;
  unsigned int state_count = 0;

  state_count++;
  state_sequence = (unsigned int*)ck_realloc(state_sequence, state_count*sizeof(unsigned int));
  state_sequence[state_count-1] = 0; //function code: 0-255

  if (buf == NULL || buf_size == 0)
    goto RET;

  while (cur_ptr <= end_ptr){

    unsigned int remaining_buf_size = end_ptr - cur_ptr + 1;
    // mbap + func id =8
    if (remaining_buf_size >= sizeof(mbap_be)){
      mbap_be *header = (mbap_be *)cur_ptr;
      //function code '0' is not valid，1-255，128-255 for exception response
      //normal response: function code (1-127) + data response
      //exception response: exception function code (function code + 0x80) + exception code (1 Byte, )
      unsigned int message_code = header->fid;

      unsigned short remaining_packet_length = ushort_be_to_se(header->length);
      unsigned short data_length = remaining_packet_length - 2;
      unsigned int packet_length = remaining_packet_length + 6;
      unsigned short available_data_length = (remaining_buf_size > packet_length) ? data_length : remaining_buf_size - sizeof(mbap_be);

      cur_ptr = cur_ptr + sizeof(mbap_be);
      state_count++;
      state_sequence = (unsigned int *)ck_realloc(state_sequence, state_count*sizeof(unsigned int));
      //exception response: exception function code (function code + 0x80) + exception code (1 Byte, )
      if (header->fid > 127 || header->fid == 0){ //exception response
        if (available_data_length > 0){
          unsigned int len = (available_data_length > 3)? 3: available_data_length; //exception code 1 byte, why 3 ?
          memcpy((char *)&message_code+1, cur_ptr, len);
        }else
          break;
      }
      state_sequence[state_count-1] = message_code;
      cur_ptr = cur_ptr + available_data_length;
      for(int i = sizeof(unsigned int)-1; i>=0; i--){
        unsigned char byte = (message_code >> (i*8)) & 0xFF;
      }
    } else
        break;
  }

  RET:
    *state_count_ref = state_count;
    return state_sequence;
}

unsigned int *extract_response_codes_iec104(unsigned char *buf, unsigned int buf_size, unsigned int *state_count_ref) {
    unsigned char *cur_ptr = buf;
    unsigned char *end_ptr = buf + buf_size;
    unsigned int *state_sequence = NULL;
    unsigned int state_count = 0;
    unsigned char start_byte = 0x68;
    unsigned short length = 0;
    unsigned int remaining_buf = 0;
    unsigned int message_code;
    unsigned char code_temp;

    state_count++;
    state_sequence = (unsigned int *)ck_realloc(state_sequence, state_count * sizeof(unsigned int));
    state_sequence[state_count - 1] = 0;

    //state_count++;
    //state_sequence = (unsigned int *)ck_realloc(state_sequence, state_count * sizeof(unsigned int));
    //state_sequence[state_count - 1] = UINT_MAX; // state including 0

    if (buf == NULL || buf_size == 0) {
        *state_count_ref = state_count;
        return state_sequence;
    }

    while (cur_ptr < end_ptr) {
        remaining_buf = end_ptr - cur_ptr;
        // Check if the first bytes are 0x68
        if (remaining_buf < 3) {
            break;
        }
        if (*cur_ptr == start_byte) {
          code_temp = *(cur_ptr + 2);
          if ((code_temp & 0x01) == 0) { // I-format: message code = control-field1 +Type Identification
            message_code = 0;
            if (cur_ptr + 6 < end_ptr) {
              memcpy((char *)&message_code + 1, cur_ptr + 6, 1);
            }
          } else if ((code_temp & 0x03) == 1) { // S-format
            message_code = code_temp & 0x03;
          } else if ((code_temp & 0x03) == 3) { // U-format
            message_code = code_temp;
          } else {
            message_code = code_temp;
          }
          state_count++;
          state_sequence = (unsigned int *)ck_realloc(
              state_sequence, state_count * sizeof(unsigned int));
          state_sequence[state_count - 1] = message_code;

          length = *(cur_ptr + 1);
          if (length + 2 <= remaining_buf) {
            cur_ptr = cur_ptr + length + 2;
          } else {
            break;
          }
        } else {
          cur_ptr++;
        }
    }

    *state_count_ref = state_count;
    return state_sequence;
}

unsigned int *extract_response_codes_ethernetip(unsigned char *buf, unsigned int buf_size, unsigned int *state_count_ref) {
  unsigned char *cur_ptr = buf;
  unsigned char *end_ptr = buf + buf_size;
  unsigned int *state_sequence = NULL;
  unsigned int state_count = 0;
  unsigned int remaining_buf = 0;
  unsigned int length = 0;
  unsigned int status_code = 0;
  unsigned int command_code = 0;
  unsigned int message_code = 0;

  state_count++;
  state_sequence = (unsigned int *)ck_realloc(state_sequence, state_count * sizeof(unsigned int));
  state_sequence[state_count - 1] = 0;

  if (buf == NULL || buf_size == 0) {
    *state_count_ref = state_count;
    return state_sequence;
  }

  while (cur_ptr < end_ptr) {
    remaining_buf = end_ptr - cur_ptr;

    if (remaining_buf < 24) {
      break;
    }
    //command code: 2 bytes, little-endian
	//example: ListServices 04 00, command code =  0x0004
    command_code = cur_ptr[0] | (cur_ptr[1] << 8);
    //status field is from bytes 8-11, little-endian
    status_code = (cur_ptr[11] << 24) | (cur_ptr[10] << 16) | (cur_ptr[9] << 8) | cur_ptr[8];
    message_code = command_code + status_code;
    
    state_count++;
    state_sequence = (unsigned int *)ck_realloc(state_sequence, state_count * sizeof(unsigned int));
    state_sequence[state_count - 1] = message_code;

    length = (cur_ptr[3] << 8) | cur_ptr[2];
	
    if (length >= 0 && remaining_buf >= 24 + length) {
      cur_ptr = cur_ptr + 24 +length;
    } else {
      break;
    }
  }

  *state_count_ref = state_count;
  return state_sequence;
}

//SLMP ASCII code
unsigned int *extract_response_codes_slmpa(unsigned char *buf, unsigned int buf_size, unsigned int *state_count_ref) {
  unsigned char *cur_ptr = buf;
  unsigned char *cur = buf;  //point to beginning of response message
  unsigned char *end_ptr = buf + buf_size;
  unsigned int *state_sequence = NULL;
  unsigned int state_count = 0;
  unsigned int remaining_buf = 0;
  unsigned long length = 0;
  char message_code_str[5];
  unsigned int message_code = 0;

  state_count++;
  state_sequence = (unsigned int *)ck_realloc(state_sequence, state_count * sizeof(unsigned int));
  state_sequence[state_count - 1] = 0;

  if (buf == NULL || buf_size == 0) {
    *state_count_ref = state_count;
    return state_sequence;
  }

  unsigned char start_byte[] = {0x44, 0x34, 0x30, 0x30};
  unsigned char start_byte_end[] = {0x30, 0x30, 0x30, 0x30};
  unsigned char start_byteN[] = {0x44, 0x30, 0x30, 0x30};
  unsigned char susscess_code[] = {0x30, 0x30, 0x30, 0x30};
  unsigned char error_code[] = {0x30, 0x34, 0x30, 0x30};
  unsigned short subheader_length = 0;

  while (cur_ptr < end_ptr -12) {
    remaining_buf = end_ptr - cur_ptr;

    if(memcmp(cur_ptr, start_byte, 4) == 0){
      if(memcmp(cur_ptr+8, start_byte_end, 4) == 0){
        subheader_length = 12;
      }else{
        cur_ptr++;
        cur++;
        continue;
      }
    }else if(memcmp(cur_ptr, start_byteN, 4) == 0){
      subheader_length = 4;
    }else{
      cur_ptr++;
      cur++;
      continue;
    }

    if (remaining_buf < 14 + subheader_length) {
      *state_count_ref = state_count;
      return state_sequence;
    }

    // Get length from ASCII hex chars
    cur_ptr = cur_ptr + subheader_length + 10;  //point to beginning of length field
    char len_str[5];
    memcpy(len_str, cur_ptr, 4);
    len_str[4] = '\0';
    length = strtol(len_str, NULL, 16);
    if(length <= 0)
      break;

    if(length >= 4 && remaining_buf > 14 + subheader_length + length) {
      cur_ptr = cur_ptr + 4; //point to beginning of End code field
      if(memcmp(cur_ptr, susscess_code, 4) == 0){   //message code = end code (0000)
        message_code = 0;
      }else if(memcmp(cur_ptr, error_code, 4) == 0){  //message code = command code 
        if(cur_ptr + 18 < end_ptr){
          cur_ptr = cur_ptr + 4; //point to beginning of error information field
          cur_ptr = cur_ptr + 10; //point to beginning of command field
          memcpy(message_code_str, cur_ptr, 4);
          message_code_str[4] = '\0';
          message_code = strtol(message_code_str, NULL, 16);
        }else{
          message_code = 0x0400;  //message code = end code (0400)
        }
      }else{
        memcpy(message_code_str, cur_ptr, 4);   //message code is the end code field
        message_code_str[4] = '\0';
        message_code = strtol(message_code_str, NULL, 16);
      }
    } else {
      cur_ptr = cur_ptr + 4; //point to beginning of End code field
      size_t len = remaining_buf - 14 - subheader_length;
      len = (len > 4)? 4: len;
      memcpy(message_code_str, cur_ptr, len);
      message_code_str[len] = '\0';
      message_code = strtol(message_code_str, NULL, 16);
    }
    
    state_count++;
    state_sequence = (unsigned int *)ck_realloc(state_sequence, state_count * sizeof(unsigned int));
    state_sequence[state_count - 1] = message_code;

    if (remaining_buf > 14 + subheader_length + length) {
      cur = cur + 14 + subheader_length + length;
      cur_ptr = cur;
    } else {
      break;
    }
  }

  *state_count_ref = state_count;
  return state_sequence;
}

//SLMP Binary code
unsigned int *extract_response_codes_slmpb(unsigned char *buf, unsigned int buf_size, unsigned int *state_count_ref) {
  unsigned char *cur_ptr = buf;
  unsigned char *cur = buf;  //point to beginning of response message
  unsigned char *end_ptr = buf + buf_size;
  unsigned int *state_sequence = NULL;
  unsigned int state_count = 0;
  unsigned int remaining_buf = 0;
  unsigned long length = 0;
  unsigned int message_code = 0;

  state_count++;
  state_sequence = (unsigned int *)ck_realloc(state_sequence, state_count * sizeof(unsigned int));
  state_sequence[state_count - 1] = 0;

  if (buf == NULL || buf_size == 0) {
    *state_count_ref = state_count;
    return state_sequence;
  }

  unsigned char start_byte[] = {0xd4, 0x00};
  unsigned char start_byte_end[] = {0x00, 0x00};
  unsigned char start_byteN[] = {0xd0, 0x00};
  unsigned char susscess_code[] = {0x00, 0x00};
  unsigned char error_code[] = {0x00, 0x04};
  unsigned short subheader_length = 0;

  while (cur_ptr < end_ptr - 6) {
    remaining_buf = end_ptr - cur_ptr;

    if(memcmp(cur_ptr, start_byte, 2) == 0){
      if(memcmp(cur_ptr+4, start_byte_end, 2) == 0){
        subheader_length = 6;
      }else{
        cur_ptr++;
        cur++;
        continue;
      }
    }else if(memcmp(cur_ptr, start_byteN, 2) == 0){
      subheader_length = 2;
    }else{
      cur_ptr++;
      cur++;
      continue;
    }

    if (remaining_buf < 7 + subheader_length) {
      *state_count_ref = state_count;
      return state_sequence;
    }

    // Get length from ASCII hex chars
    cur_ptr = cur_ptr + subheader_length + 5;  //point to beginning of length field
    length = (cur_ptr[1] << 8) | cur_ptr[0];
    if(length <= 0)
      break;

    if(length >= 2 && remaining_buf > 7 + subheader_length + length) {
      cur_ptr = cur_ptr + 2; //point to beginning of End code field
      if(memcmp(cur_ptr, susscess_code, 2) == 0){   //message code = end code (0000)
        message_code = 0;
      }else if(memcmp(cur_ptr, error_code, 2) == 0){  //message code = command code 
        if(cur_ptr + 9 < end_ptr){
          cur_ptr = cur_ptr + 2; //point to beginning of error information field
          cur_ptr = cur_ptr + 5; //point to beginning of command field
          message_code = (cur_ptr[1] << 8) | cur_ptr[0];
        }else{
          message_code = 0x0400;   //message code = end code (0400)
        }
      }else{
        message_code = (cur_ptr[1] << 8) | cur_ptr[0]; //message code = end code , and the end code is fault code
      }
    } else {
      size_t len = remaining_buf - 7 - subheader_length;
      cur_ptr = cur_ptr + 2; //point to beginning of End code field
      message_code = (len > 2)? ((cur_ptr[1] << 8) | cur_ptr[0]): cur_ptr[0];
    }
    
    state_count++;
    state_sequence = (unsigned int *)ck_realloc(state_sequence, state_count * sizeof(unsigned int));
    state_sequence[state_count - 1] = message_code;

    if (remaining_buf > 7 + subheader_length + length) {
      cur = cur + 7 + subheader_length + length;
      cur_ptr = cur;
    } else {
      break;
    }
  }

  *state_count_ref = state_count;
  return state_sequence;
}

//OPC UA Connection Protocol
unsigned int *extract_response_codes_opcuacp(unsigned char *buf, unsigned int buf_size, unsigned int *state_count_ref) {
  unsigned char *cur_ptr = buf;
  unsigned char *end_ptr = buf + buf_size;
  unsigned int *state_sequence = NULL;
  unsigned int state_count = 0;
  unsigned int remaining_buf = 0;
  unsigned long length = 0;
  unsigned int message_code = 0;

  state_count++;
  state_sequence = (unsigned int *)ck_realloc(state_sequence, state_count * sizeof(unsigned int));
  state_sequence[state_count - 1] = 0;

  if (buf == NULL || buf_size == 0) {
    *state_count_ref = state_count;
    return state_sequence;
  }

  const char* messagetype[]={
    "MSG",
    "OPN", 
    "CLO",
    "HEL",
    "ERR",
    "ACK",
    "RHE",
  };

  char msg_type[4] = {0};
  int flag = 0; //0: not found; 1: found

  while (cur_ptr <= end_ptr - 8) { 

    remaining_buf = end_ptr - cur_ptr;
    flag = 0;

    memcpy(msg_type, cur_ptr, 3);
    msg_type[3] = '\0';
    for (int i = 0; i < sizeof(messagetype) / sizeof(messagetype[0]); i++) {
      if (strncmp(msg_type, messagetype[i], 3) == 0) {
        flag = 1;
        // Convert 3 chars to integer by treating as 24-bit number
        message_code = (messagetype[i][0] << 16) | (messagetype[i][1] << 8) | messagetype[i][2];
        break;
      }
    }

    if(flag == 0){
      cur_ptr++;
      continue;
    }

    // Extract length from bytes 4-7 (little-endian)
    length = (cur_ptr[7] << 24) | (cur_ptr[6] << 16) | (cur_ptr[5] << 8) | cur_ptr[4];
    if (message_code == ('E' << 16 | 'R' << 8 | 'R') && length >= 12) {  //message code = ERR
      unsigned int error_code;
      memcpy(&error_code, cur_ptr+8, 4);
      message_code = error_code;
    }

    state_count++;
    state_sequence = (unsigned int *)ck_realloc(state_sequence, state_count * sizeof(unsigned int));
    state_sequence[state_count - 1] = message_code;

    cur_ptr = cur_ptr + length;
  }

  *state_count_ref = state_count;
  return state_sequence;
}

unsigned int *extract_response_codes_dnp3(unsigned char *buf, unsigned int buf_size, unsigned int *state_count_ref) {
  unsigned char *cur_ptr = buf;
  unsigned char *end_ptr = buf + buf_size;
  unsigned int *state_sequence = NULL;
  unsigned int state_count = 0;
  unsigned short length = 0;
  unsigned int remaining_buf = 0;
  unsigned int message_code;
  unsigned char start_byte[] = {0x05, 0x64};
  unsigned short data_field_number = 0;
  unsigned short remaining_data = 0;
  unsigned int message_length = 0;

  state_count++;
  state_sequence = (unsigned int *)ck_realloc(state_sequence, state_count * sizeof(unsigned int));
  state_sequence[state_count - 1] = 0;

  //state_count++;
  //state_sequence = (unsigned int *)ck_realloc(state_sequence, state_count * sizeof(unsigned int));
  //state_sequence[state_count - 1] = UINT_MAX; // state including 0

  if (buf == NULL || buf_size == 0) {
    *state_count_ref = state_count;
    return state_sequence;
  }

  while (cur_ptr < end_ptr) {
    remaining_buf = end_ptr - cur_ptr;

    if(remaining_buf < 2 || memcmp(cur_ptr, start_byte, 2) != 0){
      cur_ptr++;
      continue;
    }

    // Check if enough bytes for minimum DNP3 header 
    if (remaining_buf >= 8) {
      length = cur_ptr[2];   //length is in bytes 3-4
      if(length < 5){
        cur_ptr++;
        continue;
      }

      message_code = cur_ptr[3];

      data_field_number = (length - 5)/16;
      remaining_data = length - 5 - data_field_number*16;
      message_length = 2 + length + 2 + data_field_number*2 + (remaining_data > 0 ? 2 : 0);
    
      state_count++;
      state_sequence = (unsigned int *)ck_realloc(state_sequence, state_count * sizeof(unsigned int));
      state_sequence[state_count - 1] = message_code;

      cur_ptr = cur_ptr + message_length;
    }else{
      break;
    }
  }

  *state_count_ref = state_count;
  return state_sequence;
}

unsigned int *extract_response_codes_bacnetip(unsigned char *buf, unsigned int buf_size, unsigned int *state_count_ref) {
  unsigned char *cur_ptr = buf;
  unsigned char *end_ptr = buf + buf_size;
  unsigned int *state_sequence = NULL;
  unsigned int state_count = 0;
  unsigned char start_byte = 0x81;
  unsigned short length = 0;
  unsigned int remaining_buf = 0;
  unsigned int message_code;

  state_count++;
  state_sequence = (unsigned int *)ck_realloc(state_sequence, state_count * sizeof(unsigned int));
  state_sequence[state_count - 1] = 0;

  //state_count++;
  //state_sequence = (unsigned int *)ck_realloc(state_sequence, state_count * sizeof(unsigned int));
  //state_sequence[state_count - 1] = UINT_MAX; // state including 0

  if (buf == NULL || buf_size == 0) {
    *state_count_ref = state_count;
    return state_sequence;
  }

  while (cur_ptr < end_ptr) {
    remaining_buf = end_ptr - cur_ptr;
    // Check if the first bytes are 0x81
    if (*cur_ptr != start_byte) {
      cur_ptr++;
      continue;
    }

    if(remaining_buf < 2)
      break;

    message_code = cur_ptr[1];
    state_count++;
    state_sequence = (unsigned int *)ck_realloc(state_sequence, state_count * sizeof(unsigned int));
    state_sequence[state_count - 1] = message_code;

    if (remaining_buf >= 4) { 
      length = (cur_ptr[2] << 8) | cur_ptr[3];
        // Add bounds checking to prevent excessive memory allocation
        // BACnet/IP length field should be reasonable (4 bytes to 1476 bytes typically)
        if (length < 4 || length > 65535 || length > remaining_buf) {
            // Invalid length field, skip this malformed packet
            cur_ptr++;
            continue;
        }
      if(remaining_buf >= length){
        cur_ptr = cur_ptr + length;
      } else {
        break;
      }
    }else
      break; 
  }

  *state_count_ref = state_count;
  return state_sequence;
}

unsigned int* extract_response_codes_mqtt(unsigned char* buf, unsigned int buf_size, unsigned int* state_count_ref)
{
  unsigned char *mem;
	unsigned int byte_count = 0;
	unsigned int mem_count = 0;
	unsigned int mem_size = 1024;
	unsigned int *state_sequence = NULL;
	unsigned int state_count = 0;
  // Packet headers for MQTT broker responses
	char start1[1]={0x20}; // Connect Ack
	char start2[1]={0x40}; // Publish Ack
  char start3[1]={0x50}; // Publish Receive
  char start4[1]={0x62}; // Publish Release
  char start5[1]={0x70}; // Publish complete
	char start6[1]={0x90}; // Subscribe Ack
  char start7[1]={0xB0}; // Unsubscribe Ack
  char start8[1]={0xD0}; // Ping Response
  char start9[1]={0xE0}; // Disconnect
  char start10[1]={0xF0}; // Auth
	mem=(unsigned char *)ck_alloc(mem_size);
	// Initial state of the response state machine
	state_count++;
	state_sequence = (unsigned int *)ck_realloc(state_sequence, state_count * sizeof(unsigned int));
	state_sequence[state_count - 1] = 0;
  while(byte_count < buf_size)
  {
    // Copy the packet header to get the message type(mem[0])
		memcpy(&mem[mem_count++], buf + byte_count++, 1);
		memcpy(&mem[mem_count], buf + byte_count++, 1);
    // printf("[fuzz]mem[0] is %02x\n",mem[0]);
    // printf("[fuzz]mem[1] is %02x\n",mem[1]);
    // Determine whether it's a response packet
    if ((mem_count > 0) && ((memcmp(&mem[0], start1, 1) == 0) || (memcmp(&mem[0], start2, 1) == 0) || (memcmp(&mem[0], start3, 1) == 0) || (memcmp(&mem[0], start4, 1) == 0) || (memcmp(&mem[0], start5, 1) == 0) || (memcmp(&mem[0], start6, 1) == 0) || (memcmp(&mem[0], start7, 1) == 0) || (memcmp(&mem[0], start8, 1) == 0) || (memcmp(&mem[0], start9, 1) == 0) || (memcmp(&mem[0], start10, 1) == 0)))
    {
      // Get the response code(message type) from the packet
      unsigned char message_code = (unsigned char)mem[0];
      // printf("[fuzz]message_code is %02x\n",message_code);
			if (message_code == 0) break;

      message_code = get_mapped_message_code(message_code);

      // Create a new state 
			state_count++;
			state_sequence = (unsigned int *)ck_realloc(state_sequence, state_count * sizeof(unsigned int));
			state_sequence[state_count - 1] = message_code;
			mem_count = 0;
      // yk
			byte_count = byte_count + mem[1];
    }
    else
    {
      mem_count++;
      if(mem_count == mem_size)
      {
        //enlarge the mem buffer
        mem_size = mem_size * 2;
        mem=(char *)ck_realloc(mem, mem_size); 
      }
    }
  }
	if (mem) ck_free(mem);
	*state_count_ref = state_count;
	return state_sequence;
}

// kl_messages manipulating functions

klist_t(lms) *construct_kl_messages(u8* fname, region_t *regions, u32 region_count)
{
  FILE *fseed = NULL;
  fseed = fopen(fname, "rb");
  if (fseed == NULL) PFATAL("Cannot open seed file %s", fname);

  klist_t(lms) *kl_messages = kl_init(lms);
  u32 i;

  for (i = 0; i < region_count; i++) {
    //Identify region size
    u32 len = regions[i].end_byte - regions[i].start_byte + 1;

    //Create a new message
    message_t *m = (message_t *) ck_alloc(sizeof(message_t));
    m->mdata = (char *) ck_alloc(len);
    m->msize = len;
    if (m->mdata == NULL) PFATAL("Unable to allocate memory region to store new message");
    fread(m->mdata, 1, len, fseed);

    //Insert the message to the linked list
    *kl_pushp(lms, kl_messages) = m;
  }

  if (fseed != NULL) fclose(fseed);
  return kl_messages;
}

void delete_kl_messages(klist_t(lms) *kl_messages)
{
  /* Free all messages in the list before destroying the list itself */
  message_t *m;

  int ret = kl_shift(lms, kl_messages, &m);
  while (ret == 0) {
    if (m) {
      ck_free(m->mdata);
      ck_free(m);
    }
    ret = kl_shift(lms, kl_messages, &m);
  }

  /* Finally, destroy the list */
	kl_destroy(lms, kl_messages);
}

kliter_t(lms) *get_last_message(klist_t(lms) *kl_messages)
{
  kliter_t(lms) *it;
  it = kl_begin(kl_messages);
  while (kl_next(it) != kl_end(kl_messages)) {
    it = kl_next(it);
  }
  return it;
}

u32 save_kl_messages_to_file(klist_t(lms) *kl_messages, u8 *fname, u8 replay_enabled, u32 max_count)
{
  u8 *mem = NULL;
  u32 len = 0, message_size = 0;
  kliter_t(lms) *it;

  s32 fd = open(fname, O_WRONLY | O_CREAT, 0600);
  if (fd < 0) PFATAL("Unable to create file '%s'", fname);

  u32 message_count = 0;
  //Iterate through all messages in the linked list
  for (it = kl_begin(kl_messages); it != kl_end(kl_messages) && message_count < max_count; it = kl_next(it)) {
    message_size = kl_val(it)->msize;
    if (replay_enabled) {
		  mem = (u8 *)ck_realloc(mem, 4 + len + message_size);

      //Save packet size first
      u32 *psize = (u32*)&mem[len];
      *psize = message_size;

      //Save packet content
      memcpy(&mem[len + 4], kl_val(it)->mdata, message_size);
      len = 4 + len + message_size;
    } else {
      mem = (u8 *)ck_realloc(mem, len + message_size);

      //Save packet content
      memcpy(&mem[len], kl_val(it)->mdata, message_size);
      len = len + message_size;
    }
    message_count++;
  }

  //Write everything to file & close the file
  ck_write(fd, mem, len, fname);
  close(fd);

  //Free the temporary buffer
  ck_free(mem);

  return len;
}

region_t* convert_kl_messages_to_regions(klist_t(lms) *kl_messages, u32* region_count_ref, u32 max_count)
{
  region_t *regions = NULL;
  kliter_t(lms) *it;

  u32 region_count = 1;
  s32 cur_start = 0, cur_end = 0;
  //Iterate through all messages in the linked list
  for (it = kl_begin(kl_messages); it != kl_end(kl_messages) && region_count <= max_count ; it = kl_next(it)) {
    regions = (region_t *)ck_realloc(regions, region_count * sizeof(region_t));

    cur_end = cur_start + kl_val(it)->msize - 1;
    if (cur_end < 0) PFATAL("End_byte cannot be negative");

    regions[region_count - 1].start_byte = cur_start;
    regions[region_count - 1].end_byte = cur_end;
    regions[region_count - 1].state_sequence = NULL;
    regions[region_count - 1].state_count = 0;

    cur_start = cur_end + 1;
    region_count++;
  }

  *region_count_ref = region_count - 1;
  return regions;
}

// Network communication functions

int net_send(int sockfd, struct timeval timeout, char *mem, unsigned int len) {
  unsigned int byte_count = 0;
  int n;
  struct pollfd pfd[1];
  pfd[0].fd = sockfd;
  pfd[0].events = POLLOUT;
  int rv = poll(pfd, 1, 1);

  setsockopt(sockfd, SOL_SOCKET, SO_SNDTIMEO, (char *)&timeout, sizeof(timeout));
  if (rv > 0) {
    if (pfd[0].revents & POLLOUT) {
      while (byte_count < len) {
        usleep(10);
        n = send(sockfd, &mem[byte_count], len - byte_count, MSG_NOSIGNAL);
        if (n == 0) return byte_count;
        if (n == -1) return -1;
        byte_count += n;
      }
    }
  }
  return byte_count;
}

int net_recv(int sockfd, struct timeval timeout, int poll_w, char **response_buf, unsigned int *len) {
  char temp_buf[1000];
  int n;
  struct pollfd pfd[1];
  pfd[0].fd = sockfd;
  pfd[0].events = POLLIN;
  int rv = poll(pfd, 1, poll_w);

  setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, (char *)&timeout, sizeof(timeout));
  // data received
  if (rv > 0) {
    if (pfd[0].revents & POLLIN) {
      n = recv(sockfd, temp_buf, sizeof(temp_buf), 0);
      if ((n < 0) && (errno != EAGAIN)) {
        return 1;
      }
      while (n > 0) {
        usleep(10);
        *response_buf = (unsigned char *)ck_realloc(*response_buf, *len + n + 1);
        memcpy(&(*response_buf)[*len], temp_buf, n);
        (*response_buf)[(*len) + n] = '\0';
        *len = *len + n;
        n = recv(sockfd, temp_buf, sizeof(temp_buf), 0);
        if ((n < 0) && (errno != EAGAIN)) {
          return 1;
        }
      }
    }
  } else
    if (rv < 0) // an error was returned
      return 1;

  // rv == 0 poll timeout or all data pending after poll has been received successfully
  return 0;
}

// Utility function

void save_regions_to_file(region_t *regions, unsigned int region_count, unsigned char *fname)
{
  int fd;
  FILE* fp;

  fd = open(fname, O_WRONLY | O_CREAT | O_EXCL, 0600);

  if (fd < 0) return;

  fp = fdopen(fd, "w");

  if (!fp) {
    close(fd);
    return;
  }

  int i;

  for(i=0; i < region_count; i++) {
     fprintf(fp, "Region %d - Start: %d, End: %d\n", i, regions[i].start_byte, regions[i].end_byte);
  }

  fclose(fp);
}

int str_split(char* a_str, const char* a_delim, char **result, int a_count)
{
	char *token;
	int count = 0;

	/* count number of tokens */
	/* get the first token */
	char* tmp1 = strdup(a_str);
	token = strtok(tmp1, a_delim);

	/* walk through other tokens */
	while (token != NULL)
	{
		count++;
		token = strtok(NULL, a_delim);
	}

	if (count != a_count)
	{
		return 1;
	}

	/* split input string, store tokens into result */
	count = 0;
	/* get the first token */
	token = strtok(a_str, a_delim);

	/* walk through other tokens */

	while (token != NULL)
	{
		result[count] = token;
		count++;
		token = strtok(NULL, a_delim);
	}

	free(tmp1);
	return 0;
}

void str_rtrim(char* a_str)
{
	char* ptr = a_str;
	int count = 0;
	while ((*ptr != '\n') && (*ptr != '\t') && (*ptr != ' ') && (count < strlen(a_str))) {
		ptr++;
		count++;
	}
	if (count < strlen(a_str)) {
		*ptr = '\0';
	}
}

int parse_net_config(u8* net_config, u8* protocol, u8** ip_address, u32* port)
{
  char  buf[80];
  char **tokens;
  int tokenCount = 3;

  tokens = (char**)malloc(sizeof(char*) * (tokenCount));

  if (strlen(net_config) > 80) return 1;

  strncpy(buf, net_config, strlen(net_config));
   str_rtrim(buf);

  if (!str_split(buf, "/", tokens, tokenCount))
  {
      if (!strcmp(tokens[0], "tcp:")) {
        *protocol = PRO_TCP;
      } else if (!strcmp(tokens[0], "udp:")) {
        *protocol = PRO_UDP;
      } else return 1;

      //TODO: check the format of this IP address
      *ip_address = strdup(tokens[1]);

      *port = atoi(tokens[2]);
      if (*port == 0) return 1;
  } else return 1;
  free(tokens);
  return 0;
}

u8* state_sequence_to_string(unsigned int *stateSequence, unsigned int stateCount) {
  u32 i = 0;

  u8 *out = NULL;

  char strState[STATE_STR_LEN];
  size_t len = 0;
  for (i = 0; i < stateCount; i++) {
    //Limit the loop to shorten the output string
    if ((i >= 2) && (stateSequence[i] == stateSequence[i - 1]) && (stateSequence[i] == stateSequence[i - 2])) continue;
    unsigned int stateID = stateSequence[i];
    if (i == stateCount - 1) {
      snprintf(strState, STATE_STR_LEN, "%d", (int) stateID);
    } else {
      snprintf(strState, STATE_STR_LEN, "%d-", (int) stateID);
    }
    out = (u8 *)ck_realloc(out, len + strlen(strState) + 1);
    memcpy(&out[len], strState, strlen(strState) + 1);
    len=strlen(out);
    //As Linux limit the size of the file name
    //we set a fixed upper bound here
    if (len > 150 && (i + 1 < stateCount)) {
      snprintf(strState, STATE_STR_LEN, "%s", "end-at-");
      out = (u8 *)ck_realloc(out, len + strlen(strState) + 1);
      memcpy(&out[len], strState, strlen(strState) + 1);
      len=strlen(out);

      snprintf(strState, STATE_STR_LEN, "%d", (int) stateSequence[stateCount - 1]);
      out = (u8 *)ck_realloc(out, len + strlen(strState) + 1);
      memcpy(&out[len], strState, strlen(strState) + 1);
      len=strlen(out);
      break;
    }
  }
  return out;
}


void hexdump(unsigned char *msg, unsigned char * buf, int start, int end) {
  printf("%s : ", msg);
  for (int i=start; i<=end; i++) {
    printf("%02x", buf[i]);
  }
  printf("\n");
}


u32 read_bytes_to_uint32(unsigned char* buf, unsigned int offset, int num_bytes) {
  u32 val = 0;
  for (int i=0; i<num_bytes; i++) {
    val = (val << 8) + buf[i+offset];
  }
  return val;
}
