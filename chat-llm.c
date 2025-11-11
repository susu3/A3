#define _GNU_SOURCE // asprintf
#include <stdio.h>
#include <curl/curl.h>
#include <string.h>
#include <ctype.h>
#include <dirent.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <errno.h>
#include <time.h>
#include <sys/types.h>

#include "chat-llm.h"
#include "alloc-inl.h"
#include "hash.h"
#include "aflnet.h"

// Global arrays to store binary data lengths (workaround for strlen() bug)
static size_t *sequence_lengths = NULL;
static size_t *message_lengths = NULL;
static int current_sequences_count = 0;
static int current_messages_count = 0;

// Helper function to clean up global length arrays
void cleanup_length_arrays(void) {
    if (sequence_lengths) {
        ck_free(sequence_lengths);
        sequence_lengths = NULL;
        current_sequences_count = 0;
    }
    if (message_lengths) {
        free(message_lengths);
        message_lengths = NULL;
        current_messages_count = 0;
    }
}

// Add this helper function at the top of the file
static char* json_escape_string(const char* input) {
    if (!input) return NULL;
    
    // First pass: calculate required space
    size_t len = 0;
    const char* p;
    for (p = input; *p; p++) {
        switch (*p) {
            case '\\':
            case '"':
            case '\n':
            case '\r':
            case '\t':
            case '\b':
            case '\f':
                len += 2;
                break;
            default:
                len++;
        }
    }
    
    char* output = ck_alloc(len + 1);
    char* out = output;
    
    // Second pass: escape characters
    for (p = input; *p; p++) {
        switch (*p) {
            case '\\':
                *out++ = '\\';
                *out++ = '\\';
                break;
            case '"':
                *out++ = '\\';
                *out++ = '"';
                break;
            case '\n':
                *out++ = '\\';
                *out++ = 'n';
                break;
            case '\r':
                *out++ = '\\';
                *out++ = 'r';
                break;
            case '\t':
                *out++ = '\\';
                *out++ = 't';
                break;
            case '\b':
                *out++ = '\\';
                *out++ = 'b';
                break;
            case '\f':
                *out++ = '\\';
                *out++ = 'f';
                break;
            default:
                *out++ = *p;
        }
    }
    *out = '\0';
    return output;
}

// Implementation of get_openai_api_key before it's used
const char* get_openai_api_key(void) {
    const char* token = getenv("LLM_API_KEY");
    if (!token) {
        fprintf(stderr, "Error: LLM_API_KEY environment variable not set\n");
        return NULL;
    }
    return token;
}

// -lcurl -ljson-c -lpcre2-8
// apt install libcurl4-openssl-dev libjson-c-dev libpcre2-dev libpcre2-8-0

#define MAX_TOKENS 16384   //....................????????????????????/////?
#define CONFIDENT_TIMES 3

struct MemoryStruct
{
    char *memory;
    size_t size;
};

//===============================================following are the functions for chat with LLM===============================================
//contents: the response come from the LLM, size: the size of the data, nmemb: the number of the data, userp: the pointer to the struct MemoryStruct to store the data
static size_t chat_with_llm_helper(void *contents, size_t size, size_t nmemb, void *userp)
{
    size_t realsize = size * nmemb; //total size of the response
    struct MemoryStruct *mem = (struct MemoryStruct *)userp;

    mem->memory = realloc(mem->memory, mem->size + realsize + 1);
    if (mem->memory == NULL)
    {
        /* out of memory! */
        printf("not enough memory (realloc returned NULL)\n");
        return 0;
    }

    memcpy(&(mem->memory[mem->size]), contents, realsize);
    mem->size += realsize;
    mem->memory[mem->size] = 0; //ensure the string is terminated

    return realsize;
}

//get the "content" from the curl response
char *chat_with_llm(char *prompt, int tries, float temperature)
{
    CURL *curl;
    CURLcode res = CURLE_OK;
    char *answer = NULL;
    //char *url = "https://api.openai.com/v1/chat/completions";
    char *url = "https://openrouter.ai/api/v1/chat/completions";
    const char* api_key = get_openai_api_key();
    if (!api_key)
    {
        FATAL("Error: LLM_API_KEY environment variable not set");
        return NULL;
    }

    char *auth_header;
    if (asprintf(&auth_header, "Authorization: Bearer %s", api_key) == -1) {
        printf("Error: Failed to allocate memory for authorization header\n");
        return NULL;
    }
    char *content_header = "Content-Type: application/json";
    char *accept_header = "Accept: application/json";
    char *data = NULL;
    if (prompt == NULL) {
        FATAL("Error: prompt is NULL");
        return NULL;
    }

    // Replace the current approach with a more robust JSON construction method
    // using json-c library to properly format JSON objects
    json_object *json_payload = json_object_new_object();
    json_object *json_messages;
    
    // Parse the messages array from the prompt
    json_messages = json_tokener_parse(prompt);
    if (json_messages == NULL) {
        // If parsing failed, the prompt is not a valid JSON array
        // Create a simple message object as fallback
        json_messages = json_object_new_array();
        json_object *message = json_object_new_object();
        json_object *content = json_object_new_string(prompt);
        json_object_object_add(message, "role", json_object_new_string("user"));
        json_object_object_add(message, "content", content);
        json_object_array_add(json_messages, message);
    }
    
    // Add all properties to the request
    json_object_object_add(json_payload, "model", json_object_new_string("google/gemini-2.5-pro"));
    json_object_object_add(json_payload, "messages", json_messages);
    json_object_object_add(json_payload, "max_tokens", json_object_new_int(MAX_TOKENS));
    json_object_object_add(json_payload, "temperature", json_object_new_double(temperature));
    
    // Get the JSON string
    const char *json_str = json_object_to_json_string_ext(json_payload, JSON_C_TO_STRING_PLAIN);
    data = strdup(json_str);
    
    // Add debug output to see the actual request payload
    //printf("Debug - API Request payload: %s\n", data);
    
    curl_global_init(CURL_GLOBAL_DEFAULT);
    do
    {
        struct MemoryStruct chunk;

        chunk.memory = malloc(1); /* will be grown as needed by the realloc above */
        chunk.size = 0;           /* no data at this point */

        curl = curl_easy_init();
        if (curl)
        {
            struct curl_slist *headers = NULL;
            headers = curl_slist_append(headers, auth_header);
            headers = curl_slist_append(headers, content_header);
            headers = curl_slist_append(headers, accept_header);

            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data);
            curl_easy_setopt(curl, CURLOPT_URL, url);
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, chat_with_llm_helper);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&chunk);

            printf("Sending request to LLM\n");
            res = curl_easy_perform(curl);

            if (res == CURLE_OK)
            {
                json_object *jobj = json_tokener_parse(chunk.memory);

                // Check if the "choices" key exists
                if (json_object_object_get_ex(jobj, "choices", NULL))
                {
                    json_object *choices = json_object_object_get(jobj, "choices");
                    json_object *first_choice = json_object_array_get_idx(choices, 0); //the first element of the array choices
                    const char *data;

                    // The answer begins with a newline character, so we remove it
                    json_object *jobj4 = json_object_object_get(first_choice, "message");
                    json_object *jobj5 = json_object_object_get(jobj4, "content");
                    data = json_object_get_string(jobj5);
                    if (data[0] == '\n')
                        data++;
                    answer = strdup(data); //copy data to answer
                }
                else
                {
                    // Log request details
                    fprintf(stderr, "LLM API error. Request details:\n");
                    fprintf(stderr, "URL: %s\n", url);
                    //fprintf(stderr, "Headers:\n%s\n%s\n%s\n", auth_header, content_header, accept_header);
                    //fprintf(stderr, "Request body: %s\n", data);
                    fprintf(stderr, "Response: %s\n", chunk.memory);
                }
                json_object_put(jobj); //free memory
            }
            else
            {
                fprintf(stderr, "LLM API error: %s\n", curl_easy_strerror(res));
            }

            curl_slist_free_all(headers); 
            curl_easy_cleanup(curl);
        }

        free(chunk.memory);
    } while ((res != CURLE_OK || answer == NULL) && (--tries > 0));

    if (data != NULL)
    {
        free(data);
    }

    curl_global_cleanup();
    return answer;
}

//===============================================following are the functions for generate initial seeds===============================================

//construct the prompt for the generate initial seeds, according to the provided RFC document and some examples
// Function: construct_prompt_for_seeds
// Parameters:
//   - protocol_name: Name of the protocol (e.g., "MODBUS", "IEC104")
//   - final_msg: Pointer to a char pointer that will store the final message
//   - seedfile_path: Path to the directory containing seed files
//   - rfc_path: Path to the RFC document file
// Returns:
//   A dynamically allocated string containing the formatted prompt for the LLM,
//   or NULL if an error occurs during file reading or memory allocation.
char *construct_prompt_for_seeds_message(char *protocol_name, char **final_msg, const char *seedfile_path, char *rfc_path)
{   
    char *prompt_grammars = NULL;
    char *msg = NULL;
    char *examples_str = NULL;
    char *example_seeds[2] = {NULL, NULL};
    printf("Constructing prompt for seeds message with RFC and example seed files\n");

    // Read RFC content
    // Basic path validation to prevent directory traversal
    if (strstr(rfc_path, "../") || strstr(rfc_path, "..\\")) {
        fprintf(stderr, "Error: Unsafe RFC path detected: %s\n", rfc_path);
        return NULL;
    }
    
    FILE *fp = fopen(rfc_path, "r");
    if (fp == NULL){
        fprintf(stderr, "Error opening RFC file %s\n", rfc_path);
        return NULL;
    }
    
    // Get file size safely
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long file_size = ftell(fp);
    if (file_size == -1) {
        fclose(fp);
        return NULL;
    }
    rewind(fp);

    // Allocate memory for RFC content
    char *rfc_file_content = malloc(file_size + 1);
    if (rfc_file_content == NULL){
        fprintf(stderr, "Error allocating memory for %s\n", rfc_path);
        fclose(fp);
        return NULL;
    }

    // Read file content
    size_t bytes_read = fread(rfc_file_content, 1, file_size, fp);
    fclose(fp);
    if (bytes_read != (size_t)file_size) {
        free(rfc_file_content);
        return NULL;
    }
    rfc_file_content[file_size] = '\0';

    // Load example seed files
    int example_count = 0;
    DIR *dir = opendir(seedfile_path);
    if (dir != NULL) {
        struct dirent *ent;
        while ((ent = readdir(dir)) != NULL && example_count < 2) {
            if (ent->d_type == DT_REG) {
                char *file_path;
                if (asprintf(&file_path, "%s/%s", seedfile_path, ent->d_name) == -1) {
                    continue;
                }
                
                FILE *seed_file = fopen(file_path, "r");
                free(file_path);
                
                if (seed_file != NULL) {
                    if (fseek(seed_file, 0, SEEK_END) == 0) {
                        long seed_size = ftell(seed_file);
                        if (seed_size > 0) {
                            rewind(seed_file);
                            example_seeds[example_count] = malloc(seed_size + 1);
                            if (example_seeds[example_count] && 
                                fread(example_seeds[example_count], 1, seed_size, seed_file) == (size_t)seed_size) {
                                example_seeds[example_count][seed_size] = '\0';
                                example_count++;
                            } else {
                                free(example_seeds[example_count]);
                            }
                        }
                    }
                    fclose(seed_file);
                }
            }
        }
        closedir(dir);
    }

    // Construct examples string with actual seed content in hex format
    if (example_count > 0) {
        // Calculate size needed for examples_str
        size_t total_size = 200; // Base text
        for (int i = 0; i < example_count; i++) {
            if (example_seeds[i]) {
                size_t seed_len = strlen(example_seeds[i]);
                // Each byte = 2 hex chars + spaces + tags + newlines
                total_size += seed_len * 3 + 200;
            }
        }
        
        examples_str = malloc(total_size);
        if (!examples_str) {
            examples_str = strdup("");
            goto cleanup;
        }
        
        // Start with header
        snprintf(examples_str, total_size, "Here are some example seed files for the %s protocol:\n", protocol_name);
        
        // Add each example seed in hex format
        for (int i = 0; i < example_count; i++) {
            if (example_seeds[i]) {
                size_t seed_len = strlen(example_seeds[i]);
                char *hex_buffer = malloc(seed_len * 3 + 100);
                if (!hex_buffer) continue;
                
                char *pos = hex_buffer;
                for (size_t j = 0; j < seed_len; j++) {
                    sprintf(pos, "%02X", (unsigned char)example_seeds[i][j]);
                    pos += 2;
                    // Add space after every 4 hex characters (2 bytes)
                    if ((j + 1) % 2 == 0 && j + 1 < seed_len) {
                        *pos++ = ' ';
                    }
                }
                *pos = '\0';
                
                // Append to examples_str
                char temp[100];
                snprintf(temp, sizeof(temp), "Example %d: <sequence>", i + 1);
                strcat(examples_str, temp);
                strcat(examples_str, hex_buffer);
                strcat(examples_str, "</sequence>\n");
                
                free(hex_buffer);
            }
        }
    } else {
        examples_str = strdup("");
    }

    if (examples_str == NULL) {
        goto cleanup;
    }

    // Construct the final message
    if (asprintf(&msg, "You are an expert in %s protocol fuzz testing. Your task is to generate **valid initial request messages** for fuzzing a binary protocol, using the **protocol specification**, writen in natural language, which is the authoritative source describing the protocol's correct structure and behavior. \n"
        "The goal is to generate as many diverse and well-formed **request messages** as possible, suitable as fuzzing seeds.\n"
        "### Output format:\n"
        "- Wrap each request example with <sequence></sequence>.\n"
        "- Inside each sequence, there should be one space after every 4 hex characters. (i.e., group 2 bytes = 4 hex digits).\n"
        "- The output must be **hex-encoded and reflect the actual binary request message**.\n"
        "- Ensure the **packet length field (if present)** matches the true length of the remaining data, as defined in the grammar.\n"
        "- Do not include field names, comments, or explanations — just the formatted sequences.\n"
        "### Example (format only):\n"
        "For the MODBUS protocol, a request sequence example is (in hex): <sequence>0000 0000 0008 FF16 0004 00F2 0025</sequence>\n"
        "For the IEC104 protocol, a request sequence example is (in hex): <sequence>6804 0700 0000 6804 4300 0000 6804 1300 0000</sequence>\n"
        "%s"
        "The Specification Document is as follows:\n"
        "=== BEGIN SPEC ===\n"
        "%s\n"
        "=== END SPEC ===\n",
        protocol_name, examples_str, rfc_file_content) == -1) {
        goto cleanup;
    }

    *final_msg = msg;

    // Create the final prompt_grammars with proper escaping
    char* escaped_msg = json_escape_string(msg);
    if (!escaped_msg) {
        goto cleanup;
    }
    
    if (asprintf(&prompt_grammars, "[{\"role\": \"user\", \"content\": \"%s\"}]", escaped_msg) == -1) {
        prompt_grammars = NULL;
    }
    
    ck_free(escaped_msg);

cleanup:
    // Cleanup
    free(rfc_file_content);
    free(examples_str);
    for (int i = 0; i < 2; i++) {
        free(example_seeds[i]);
    }
    // Don't free msg here as it's assigned to *final_msg and caller expects to use it
    // Only free msg if we're returning NULL (error case)
    if (!prompt_grammars) {
        free(msg);
    }

    return prompt_grammars;
}

char *construct_prompt_for_seeds_sequence(char *protocol_name, char **final_msg, char *rfc_path){
    char *msg = NULL;
    printf("Constructing prompt for seeds sequence with RFC\n");

    // Read RFC content
    // Basic path validation to prevent directory traversal
    if (strstr(rfc_path, "../") || strstr(rfc_path, "..\\")) {
        fprintf(stderr, "Error: Unsafe RFC path detected: %s\n", rfc_path);
        return NULL;
    }
    
    FILE *fp = fopen(rfc_path, "r");
    if (fp == NULL){
        fprintf(stderr, "Error opening RFC file %s\n", rfc_path);
        return NULL;
    }
    
    // Get file size safely
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long file_size = ftell(fp);
    if (file_size == -1) {
        fclose(fp);
        return NULL;
    }
    rewind(fp);

    // Allocate memory for RFC content
    char *rfc_file_content = malloc(file_size + 1);
    if (rfc_file_content == NULL){
        fprintf(stderr, "Error allocating memory for %s\n", rfc_path);
        fclose(fp);
        return NULL;
    }

    // Read file content
    size_t bytes_read = fread(rfc_file_content, 1, file_size, fp);
    fclose(fp);
    if (bytes_read != (size_t)file_size) {
        free(rfc_file_content);
        return NULL;
    }
    rfc_file_content[file_size] = '\0';

    // First, construct the message content (not yet JSON)
    char *content = NULL;
    if (asprintf(&content, "You are an expert in %s protocol fuzz testing. Your task is to generate valid client-side request message sequences for an industrial binary protocol using the Specification Document provided below. A request sequence is a series of multiple client-side request messages, sent one after another to follow the protocol's client state machine and message grammar. Each request message must be encoded in hexadecimal and strictly follow the inferred grammar. The full sequence should simulate a realistic multi-step client session and MUST omit any server-side responses.\n\n"
        "# Inputs (authoritative):\n"
        "- The protocol specification written in natural language. Use it to reconstruct the protocol's client message grammar and a minimal but sufficient client-side state machine. Resolve ambiguities by looking for examples, field descriptions, and constraints in the spec.\n\n"
        "# What you must do (silently, no extra text in output):\n"
        "1) From the Specification Document, silently reconstruct a client-side state machine (states, allowed client requests, transitions) and a message grammar (fields, sizes, types, constraints, computed fields like length/CRC, alignment/padding rules).\n"
        "2) Validate that each planned path uses CLIENT REQUESTS ONLY (no server frames) and is a valid path through the inferred client state machine.\n"
        "3) For each request message you will output: apply all field constraints; compute derived fields (length, sequence numbers, checksums/CRCs) correctly; and ensure inter-message dependencies (handles, session IDs, counters) are consistent across the sequence.\n"
        "4) Encoding rules:\n"
        "   - Output message bytes in hexadecimal. Insert a space every 4 hex digits (e.g., \"AABB CCDD ...\").\n"
        "   - Follow the endianness implied by the spec. If the spec is silent and no examples imply otherwise, default to network byte order (big-endian) and be consistent.\n"
        "   - Pad ASCII strings with null bytes if a fixed length is required.\n"
        "   - If a field is optional per the spec, include or omit it according to the chosen valid path.\n"
        "5) Output format (STRICT):\n"
        "<sequence>\n"
        "  <message>...</message>\n"
        "  <message>...</message>\n"
        "</sequence>\n"
        "- DO NOT output any explanations, assumptions, or comments—ONLY the sequences in the exact tags above.\n\n"
        "# Diversity & Coverage:\n"
        "- Generate as many diverse, valid client-only sequences as possible, each corresponding to a distinct valid path.\n"
        "- Vary legal field values, optional sections, and multi-step flows while respecting all constraints and cross-message dependencies. Avoid superficial repetition.\n\n"
        "### Example (format only):\n"
        "For the IEC104 protocol, an example sequence format is: <sequence><message>6804 0700 0000</message><message>6804 4300 0000</message><message>6804 1300 0000</message></sequence>\n\n"
        "The Specification Document is as follows:\n"
        "=== BEGIN SPEC ===\n"
        "%s\n"
        "=== END SPEC ===\n", 
        protocol_name, rfc_file_content) == -1) {
        FATAL("Failed to construct seeds sequence content");
        goto cleanup;
    }

    *final_msg = content;

    // Now escape the content for JSON and construct the JSON message
    char* escaped_content = json_escape_string(content);
    if (!escaped_content) {
        goto cleanup;
    }
    
    if (asprintf(&msg, "[{\"role\": \"user\", \"content\": \"%s\"}]", escaped_content) == -1) {
        ck_free(escaped_content);
        msg = NULL;
        goto cleanup;
    }
    
    ck_free(escaped_content);
    
    return msg;

cleanup:
    // Cleanup
    free(rfc_file_content);
    // Don't free content here as it's assigned to *final_msg and caller expects to use it
    // Only free content and msg if we're returning NULL (error case)
    if (!msg) {
        free(content);
    }
    return NULL;
}

// extract the messages from the sequences
// Function: extract_messages
// Parameters:
//   - sequences_answer: the LLM output
//   - num_messages: the number of sequences extracted from the LLM output
// Returns:
//   - the messages extracted from the LLM output
char** extract_messages(const char* sequences_answer, int* num_messages) {
    if (!sequences_answer || !num_messages) {
        return NULL;
    }
    
    *num_messages = 0;
    char** messages = NULL;
    
    const char* sequence_start = "<sequence>";
    const char* sequence_end = "</sequence>";
    const char* message_start = "<message>";
    const char* message_end = "</message>";
    
    const char* seq_pos = sequences_answer;
    
    // Find each <sequence>...</sequence> block
    while ((seq_pos = strstr(seq_pos, sequence_start)) != NULL) {
        seq_pos += strlen(sequence_start);
        const char* seq_end = strstr(seq_pos, sequence_end);
        if (!seq_end) break;
        
        // Extract content within this sequence
        size_t seq_len = seq_end - seq_pos;
        char* sequence_content = malloc(seq_len + 1);
        if (!sequence_content) {
            PFATAL("Failed to allocate memory for sequence content");
        }
        strncpy(sequence_content, seq_pos, seq_len);
        sequence_content[seq_len] = '\0';
        
        // Now extract all messages within this sequence and concatenate them
        const char* msg_pos = sequence_content;
        char* combined_message = malloc(seq_len * 2 + 1); // Extra space for safety
        if (!combined_message) {
            free(sequence_content);
            PFATAL("Failed to allocate memory for combined message");
        }
        combined_message[0] = '\0';
        size_t combined_len = 0;
        size_t combined_capacity = seq_len * 2;
        
        // Extract and concatenate all messages in this sequence
        while ((msg_pos = strstr(msg_pos, message_start)) != NULL) {
            msg_pos += strlen(message_start);
            const char* msg_end = strstr(msg_pos, message_end);
            if (!msg_end) break;
            
            size_t msg_len = msg_end - msg_pos;
            char* message_content = malloc(msg_len + 1);
            if (!message_content) {
                free(sequence_content);
                free(combined_message);
                PFATAL("Failed to allocate memory for message content");
            }
            strncpy(message_content, msg_pos, msg_len);
            message_content[msg_len] = '\0';
            
            // Remove whitespace and validate hex
            char* clean_message = malloc(msg_len + 1);
            if (!clean_message) {
                free(sequence_content);
                free(combined_message);
                free(message_content);
                PFATAL("Failed to allocate memory for clean message");
            }
            
            int clean_idx = 0;
            int valid = 1;
            for (size_t i = 0; i < msg_len; i++) {
                if (isxdigit(message_content[i])) {
                    clean_message[clean_idx++] = message_content[i];
                } else if (!isspace(message_content[i])) {
                    valid = 0;
                    break;
                }
            }
            clean_message[clean_idx] = '\0';
            
            if (valid && clean_idx > 0) {
                // Append to combined message with bounds checking
                size_t clean_msg_len = strlen(clean_message);
                if (combined_len + clean_msg_len < combined_capacity) {
                    strcpy(combined_message + combined_len, clean_message);
                    combined_len += clean_msg_len;
                } else {
                    WARNF("Combined message would exceed buffer capacity, truncating");
                }
            }
            
            free(message_content);
            free(clean_message);
            msg_pos = msg_end + strlen(message_end);
        }
        
        // If we have a valid combined message, add it to results
        if (strlen(combined_message) > 0) {
            // Convert hex string to binary
            size_t hex_len = strlen(combined_message);
            if (hex_len % 2 == 0) { // Valid hex length
                size_t binary_len = hex_len / 2;
                char* binary_message = malloc(binary_len);  // No +1, no null terminator for binary data
                if (binary_message) {
                    for (size_t i = 0; i < binary_len; i++) {
                        char hex[3] = {combined_message[i*2], combined_message[i*2+1], '\0'};
                        int value = strtol(hex, NULL, 16);
                        binary_message[i] = (char)value;
                    }
                    
                    // Add to messages array
                    (*num_messages)++;
                    char** temp_messages = realloc(messages, *num_messages * sizeof(char*));
                    size_t* temp_lengths = realloc(message_lengths, *num_messages * sizeof(size_t));
                    if (!temp_messages || !temp_lengths) {
                        // Cleanup on failure
                        if (temp_messages) messages = temp_messages;
                        if (temp_lengths) message_lengths = temp_lengths;
                        free(binary_message);
                        free(sequence_content);
                        free(combined_message);
                        PFATAL("Failed to reallocate messages array or lengths array");
                    }
                    messages = temp_messages;
                    message_lengths = temp_lengths;
                    messages[*num_messages - 1] = binary_message;
                    message_lengths[*num_messages - 1] = binary_len;  // Store the actual binary length
                    current_messages_count = *num_messages;
                }
            }
        }
        
        free(sequence_content);
        free(combined_message);
        seq_pos = seq_end + strlen(sequence_end);
    }
    
    return messages;
}

// Extract sequences from LLM response
// extract the sequences from the LLM output
// Function: extract_sequences
// Parameters:
//   - seeds_answer: the LLM output
//   - num_sequences: the number of sequences extracted from the LLM output
// Returns:
//   - the sequences extracted from the LLM output
// extract_sequences function is implemented in chat-llm.c
char **extract_sequences(const char *llm_output, int *num_sequences) {
    if (!llm_output || !num_sequences) {
        return NULL;
    }
    
    *num_sequences = 0;
    const char *start_tag = "<sequence>";
    const char *end_tag = "</sequence>";
    size_t start_tag_len = strlen(start_tag);
    size_t end_tag_len = strlen(end_tag);
    
    // First pass: count the number of sequences
    const char *ptr = llm_output;
    while ((ptr = strstr(ptr, start_tag)) != NULL) {
        (*num_sequences)++;
        ptr += start_tag_len;
    }
    
    if (*num_sequences == 0) {
        return NULL;
    }
    
    // Allocate memory for sequence array and lengths array
    char **sequences = (char **)ck_alloc(*num_sequences * sizeof(char *));
    
    // Free previous length array if exists and allocate new one
    if (sequence_lengths) {
        ck_free(sequence_lengths);
    }
    sequence_lengths = (size_t *)ck_alloc(*num_sequences * sizeof(size_t));
    current_sequences_count = *num_sequences;
    
    // Second pass: extract sequences
    ptr = llm_output;
    int seq_idx = 0;
    
    while ((ptr = strstr(ptr, start_tag)) != NULL && seq_idx < *num_sequences) {
        ptr += start_tag_len;
        const char *end = strstr(ptr, end_tag);
        
        if (!end) {
            break;
        }
        
        size_t len = end - ptr;
        char *raw_seq = (char *)ck_alloc(len + 1);
        memcpy(raw_seq, ptr, len);
        raw_seq[len] = '\0';
        
        // Process sequence: remove whitespace
        char *clean_seq = (char *)ck_alloc(len + 1);
        int clean_idx = 0;
        
        for (size_t i = 0; i < len; i++) {
            if (!isspace(raw_seq[i])) {
                clean_seq[clean_idx++] = raw_seq[i];
            }
        }
        clean_seq[clean_idx] = '\0';
        
        // Convert hex string to binary
        size_t binary_len = clean_idx / 2;
        char *binary_seq = (char *)ck_alloc(binary_len);  // No +1, no null terminator for binary data
        
        for (size_t i = 0; i < binary_len; i++) {
            char hex[3] = {clean_seq[i*2], clean_seq[i*2+1], '\0'};
            int value = strtol(hex, NULL, 16);
            binary_seq[i] = (char)value;
        }
        
        sequences[seq_idx] = binary_seq;
        sequence_lengths[seq_idx] = binary_len;  // Store the actual binary length
        seq_idx++;
        
        ck_free(raw_seq);
        ck_free(clean_seq);
        
        ptr = end + end_tag_len;
    }
    
    *num_sequences = seq_idx;
    return sequences;
}

// write the sequences to the seed files
// Function: write_sequences_to_seeds
// Parameters:
//   - seedfile_path: the path to the seed file
//   - sequences: the sequences to be written to the seed file
//   - num_sequences: the number of sequences to be written to the seed file
void write_sequences_to_seeds(const char *seedfile_path, char **sequences, int num_sequences) {
  if (!seedfile_path || !sequences || num_sequences <= 0) {
    FATAL("Invalid arguments passed to write_sequences_to_seeds");
    return;
  }

  time_t t = time(NULL);
  if (t == -1) {
    PFATAL("Failed to get current time");
  }

  struct tm *tm = localtime(&t);
  if (!tm) {
    PFATAL("Failed to convert time to local time");
  }

  char timestamp[42] = {0};
  if (strftime(timestamp, sizeof(timestamp), "%Y-%m-%d-%H-%M-%S", tm) == 0) {
    PFATAL("Failed to format timestamp string");
  }

  for (int i = 0; i < num_sequences; i++) {
    if (!sequences[i]) {
      WARNF("Skipping NULL sequence at index %d", i);
      continue;
    }

    size_t filename_size = strlen(seedfile_path) + strlen(timestamp) + 32;
    char *seed_filename = malloc(filename_size);
    if (!seed_filename) {
      PFATAL("Failed to allocate memory for seed filename");
    }

    int ret = snprintf(seed_filename, filename_size, "%s/%s_%d", seedfile_path, timestamp, i + 1);
    if (ret < 0 || ret >= (int)filename_size) {
      free(seed_filename);
      PFATAL("Failed to format seed filename");
    }

    FILE *seed_file = fopen(seed_filename, "wb");
    if (seed_file == NULL) {
      free(seed_filename);
      PFATAL("Unable to create seed file '%s': %s", seed_filename, strerror(errno));
    }

    // FIXED: Use stored binary length instead of strlen() which stops at null bytes
    size_t seq_len = (i < current_sequences_count) ? sequence_lengths[i] : 0;
    if (seq_len == 0) {
        WARNF("No length information for sequence %d, skipping", i);
        fclose(seed_file);
        free(seed_filename);
        continue;
    }
    size_t written = fwrite(sequences[i], 1, seq_len, seed_file);
    if (written != seq_len) {
      fclose(seed_file);
      free(seed_filename);
      PFATAL("Failed to write to seed file '%s': %s", seed_filename, strerror(errno));
    }

    printf("Created seed file: %s (size: %zu bytes)\n", seed_filename, seq_len);

    if (fclose(seed_file) != 0) {
      free(seed_filename);
      PFATAL("Failed to close seed file '%s': %s", seed_filename, strerror(errno));
    }

    free(seed_filename);
  }
}

void write_messages_to_seeds(const char *seedfile_path, char **messages, int num_messages) {
  if (!seedfile_path || !messages || num_messages <= 0) {
    FATAL("Invalid arguments passed to write_messages_to_seeds");
    return;
  }

  time_t t = time(NULL);
  if (t == -1) {
    PFATAL("Failed to get current time");
  }

  struct tm *tm = localtime(&t);
  if (!tm) {
    PFATAL("Failed to convert time to local time");
  }

  char timestamp[42] = {0};
  if (strftime(timestamp, sizeof(timestamp), "%Y-%m-%d-%H-%M-%S", tm) == 0) {
    PFATAL("Failed to format timestamp string");
  }

  for (int i = 0; i < num_messages; i++) {
    if (!messages[i]) {
      WARNF("Skipping NULL message at index %d", i);
      continue;
    }

    size_t filename_size = strlen(seedfile_path) + strlen(timestamp) + 64;
    char *seed_filename = malloc(filename_size);
    if (!seed_filename) {
      PFATAL("Failed to allocate memory for seed filename");
    }

    int ret = snprintf(seed_filename, filename_size, "%s/seq_%s_%d", seedfile_path, timestamp, i + 1);
    if (ret < 0 || ret >= (int)filename_size) {
      free(seed_filename);
      PFATAL("Failed to format seed filename");
    }

    FILE *seed_file = fopen(seed_filename, "wb");
    if (seed_file == NULL) {
      free(seed_filename);
      PFATAL("Unable to create seed file '%s': %s", seed_filename, strerror(errno));
    }

    // FIXED: Use stored binary length instead of strlen() which stops at null bytes
    size_t msg_len = (i < current_messages_count) ? message_lengths[i] : 0;
    if (msg_len == 0) {
        WARNF("No length information for message %d, skipping", i);
        fclose(seed_file);
        free(seed_filename);
        continue;
    }
    size_t written = fwrite(messages[i], 1, msg_len, seed_file);
    if (written != msg_len) {
      fclose(seed_file);
      free(seed_filename);
      PFATAL("Failed to write to seed file '%s': %s", seed_filename, strerror(errno));
    }

    if (fclose(seed_file) != 0) {
      free(seed_filename);
      PFATAL("Failed to close seed file '%s': %s", seed_filename, strerror(errno));
    }

    printf("Created seed file: %s (size: %zu bytes)\n", seed_filename, msg_len);
    free(seed_filename);
  }
}

// test: Add this function to extract and save sequences
void extract_and_save_sequences(const char *llm_output, const char *output_dir) {
    const char *start_tag = "<sequence>";
    const char *end_tag = "</sequence>";
    const char *ptr = llm_output;
    int sequence_count = 0;

    // Create the output directory if it doesn't exist
    mkdir(output_dir, 0777);

    while ((ptr = strstr(ptr, start_tag)) != NULL) {
        ptr += strlen(start_tag);
        const char *end = strstr(ptr, end_tag);
        
        if (end == NULL) break;

        size_t len = end - ptr;
        // Add extra space for null terminator
        char *sequence = ck_alloc(len + 1);
        strncpy(sequence, ptr, len);
        sequence[len] = '\0';

        // Use ck_alloc instead of malloc for consistency
        char *cleaned_sequence = ck_alloc(len + 1);
        size_t j = 0;
        for (size_t i = 0; i < len; i++) {
            if (!isspace(sequence[i])) {
                cleaned_sequence[j++] = sequence[i];
            }
        }
        cleaned_sequence[j] = '\0';

        // Create the output file
        char filename[256];
        snprintf(filename, sizeof(filename), "%s/sequence_%d.bin", output_dir, ++sequence_count);
        FILE *fp = fopen(filename, "wb");
        
        if (fp == NULL) {
            printf("Error creating file %s\n", filename);
            ck_free(sequence);
            ck_free(cleaned_sequence);
            continue;
        }

        // Convert hex string to binary and write to file
        size_t clean_len = strlen(cleaned_sequence);
        for (size_t i = 0; i < clean_len - 1; i += 2) {
            char hex[3] = {cleaned_sequence[i], cleaned_sequence[i+1], '\0'};
            int value = strtol(hex, NULL, 16);
            fputc(value, fp);
        }

        fclose(fp);
        ck_free(sequence);
        ck_free(cleaned_sequence);

        ptr = end + strlen(end_tag);
    }
}

// test: Modify the write_new_seeds function to use extract_and_save_sequences
void write_new_seeds(char *enriched_file, char *contents) {
    // Create a directory for the extracted sequences
    char *sequences_dir = NULL;
    asprintf(&sequences_dir, "%s_sequences", enriched_file);

    // Extract and save sequences
    extract_and_save_sequences(contents, sequences_dir);

    free(sequences_dir);

    // Keep the original file writing logic
    FILE *fp = fopen(enriched_file, "w");
    if (fp == NULL) {
        printf("Error in opening the file %s\n", enriched_file);
        exit(1);
    }

    // remove the newline and whiltespace in the beginning of the string if any
    while (contents[0] == '\n' || contents[0] == ' ' || contents[0] == '\t' || contents[0] == '\r')
    {
        contents++;
    }

    // Check if last 4 characters of the client_request_answer string are \r\n\r\n
    // If not, add them
    int len = strlen(contents);
    if (contents[len - 1] != '\n' || contents[len - 2] != '\r' || contents[len - 3] != '\n' || contents[len - 4] != '\r')
    {
        fprintf(fp, "%s\r\n\r\n", contents);
    }
    else
    {
        fprintf(fp, "%s", contents);
    }

    fclose(fp);
}

// khash_t(strSet) * duplicate_hash(khash_t(strSet) * set)
// {
//     khash_t(strSet) *new_set = kh_init(strSet);

//     for (khiter_t k = kh_begin(set); k != kh_end(set); ++k)
//     {
//         if (kh_exist(set, k))
//         {
//             const char *val = kh_key(set, k);
//             int ret;
//             kh_put(strSet, new_set, val, &ret);
//         }
//     }

//     return new_set;
// }

// int min(int a, int b) {
//     return a < b ? a : b;
// }

// ===============================================following are the functions for protocol analysis and update===============================================
// 1. Construct prompt for message grammar analysis
char *construct_prompt_for_message_grammar(const char *protocol_name, const char *spec_path) {
    char *prompt = NULL;
    char *rfc_content = NULL;
    
    // Read RFC content
    FILE *fp = fopen(spec_path, "r");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file %s\n", spec_path);
        return NULL;
    }
    
    // Get file size safely
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long file_size = ftell(fp);
    if (file_size == -1) {
        fclose(fp);
        return NULL;
    }
    rewind(fp);

    // Allocate memory for RFC content
    rfc_content = malloc(file_size + 1);
    if (rfc_content == NULL) {
        fprintf(stderr, "Error allocating memory for %s\n", spec_path);
        fclose(fp);
        return NULL;
    }

    // Read file content
    size_t bytes_read = fread(rfc_content, 1, file_size, fp);
    fclose(fp);
    if (bytes_read != (size_t)file_size) {
        free(rfc_content);
        return NULL;
    }
    rfc_content[file_size] = '\0';

    // Construct the prompt
    if (asprintf(&prompt, 
        "[{\"role\": \"user\", \"content\": \"You are a senior industrial protocol analysis expert and you want to construct protocol request messages for protocol fuzz testing. For the %s protocol, analyze the following protocol specification and get the protocol request message grammar **running on the TCP/IP stack**. \\n"
        "The protocol request message grammar refer to the formal rules that define:\\n"
        "1. the structure and format of protocol messages (e.g., data frames, packets),\\n"
        "2. command and encoding rules,\\n"
        "3. byte-level layout and field semantics. \\n"
        "4. the dependencies between the fields, such as the length of the field is determined by the value of another field. \\n"
        "5. the value range of the field, some fields can only have specific values. \\n"
        "**Most Importantly**: only the request messages are needed to be generated! \\n"
        "The protocol request message grammar should be organized into the JSON format. Important information that cannot be organized in JSON format should be saved in a txt file in a natural language. The protocol request message grammar saved should be complete and accurate. \\n"
        "The JSON format should be like this: \\n"
        "{\\n"
        "  \\\"protocol\\\": \\\"protocol_name\\\","
        "  \\\"endian\\\": \\\"big\\\" or \\\"little\\\","
        "  \\\"messages\\\": ["
        "    {"
        "      \\\"message_name\\\": \\\"message_name\\\","
        "      \\\"fields\\\": ["
        "        {"
        "          \\\"field_name\\\": \\\"field_name\\\","
        "          \\\"field_type\\\": \\\"field_type\\\","
        "          \\\"field_length\\\": \\\"field_length\\\","
        "          \\\"dependency\\\": \\\"true\\\" or \\\"false\\\","
        "          \\\"default_value\\\": \\\"default_value\\\""
        "        }"
        "        ]"
        "      }"
        "    ]"
        "}\\n"
        "For example, the protocol request message grammar of the IEC104 protocol is as follows: \\n"
        "JSON:\\n"
        "{"
        "\\\"protocol\\\": \\\"IEC 104\\\","
        "\\\"endian\\\": \\\"big\\\","
        "\\\"messages\\\": ["
        "{"
        "\\\"message_name\\\": \\\"I-Frame\\\","
        "\\\"fields\\\": ["
        "{\\\"field_name\\\": \\\"StartByte\\\", \\\"field_type\\\": \\\"uint8\\\", \\\"field_length\\\": 1, \\\"default_value\\\": 0x68},"
        "{\\\"field_name\\\": \\\"Length\\\", \\\"field_type\\\": \\\"uint8\\\", \\\"field_length\\\": 1, \\\"dependency\\\": true},"
        "{\\\"field_name\\\": \\\"ControlField1\\\", \\\"field_type\\\": \\\"uint8\\\", \\\"field_length\\\": 1, \\\"bit_mask\\\": \\\"0xFE\\\"},"
        "{\\\"field_name\\\": \\\"ControlField2\\\", \\\"field_type\\\": \\\"uint8\\\", \\\"field_length\\\": 1},"
        "{\\\"field_name\\\": \\\"ControlField3\\\", \\\"field_type\\\": \\\"uint8\\\", \\\"field_length\\\": 1, \\\"bit_mask\\\": \\\"0xFE\\\"}," 
        "{\\\"field_name\\\": \\\"ControlField4\\\", \\\"field_type\\\": \\\"uint8\\\", \\\"field_length\\\": 1}," 
        "{\\\"field_name\\\": \\\"ASDU\\\", \\\"field_type\\\": \\\"byte[]\\\", \\\"field_length\\\": \\\"variable\\\"}"
        "]"
        "},"
        "{"
        "\\\"message_name\\\": \\\"S-Frame\\\","
        "\\\"fields\\\": ["
        "{\\\"field_name\\\": \\\"StartByte\\\", \\\"field_type\\\": \\\"uint8\\\", \\\"field_length\\\": 1, \\\"default_value\\\": 0x68},"
        "{\\\"field_name\\\": \\\"Length\\\", \\\"field_type\\\": \\\"uint8\\\", \\\"field_length\\\": 1, \\\"dependency\\\": true},"
        "{\\\"field_name\\\": \\\"ControlField1\\\", \\\"field_type\\\": \\\"uint8\\\", \\\"field_length\\\": 1, \\\"bit_mask\\\": \\\"0xFD\\\"},"
        "{\\\"field_name\\\": \\\"ControlField2\\\", \\\"field_type\\\": \\\"uint8\\\", \\\"field_length\\\": 1},"
        "{\\\"field_name\\\": \\\"ControlField3\\\", \\\"field_type\\\": \\\"uint8\\\", \\\"field_length\\\": 1, \\\"bit_mask\\\": \\\"0xFE\\\"}," 
        "{\\\"field_name\\\": \\\"ControlField4\\\", \\\"field_type\\\": \\\"uint8\\\", \\\"field_length\\\": 1},"
        "{\\\"field_name\\\": \\\"ASDU\\\", \\\"field_type\\\": \\\"byte[]\\\", \\\"field_length\\\": \\\"variable\\\"}"  
        "]"
        "},"
        "{"
        "\\\"message_name\\\": \\\"U-Frame\\\","
        "\\\"fields\\\": ["
        "{\\\"field_name\\\": \\\"StartByte\\\", \\\"field_type\\\": \\\"uint8\\\", \\\"field_length\\\": 1, \\\"default_value\\\": 0x68},"
        "{\\\"field_name\\\": \\\"Length\\\", \\\"field_type\\\": \\\"uint8\\\", \\\"field_length\\\": 1, \\\"dependency\\\": true},"
        "{\\\"field_name\\\": \\\"ControlField1\\\", \\\"field_type\\\": \\\"uint8\\\", \\\"field_length\\\": 1, \\\"bit_mask\\\": \\\"0xFF\\\"},"
        "{\\\"field_name\\\": \\\"ControlField2\\\", \\\"field_type\\\": \\\"uint8\\\", \\\"field_length\\\": 1},"
        "{\\\"field_name\\\": \\\"ControlField3\\\", \\\"field_type\\\": \\\"uint8\\\", \\\"field_length\\\": 1, \\\"bit_mask\\\": \\\"0xFE\\\"}," 
        "{\\\"field_name\\\": \\\"ControlField4\\\", \\\"field_type\\\": \\\"uint8\\\", \\\"field_length\\\": 1},"
        "{\\\"field_name\\\": \\\"ASDU\\\", \\\"field_type\\\": \\\"byte[]\\\", \\\"field_length\\\": \\\"variable\\\"}"
        "]"
        "}"
        "]"
        "}" 
        "\\n"
        "TXT:\\n"
        "The value of the field \\\"Length\\\" is the sum of the length of all the following fields, that is, the length of the field \\\"ASDU\\\" plus 4 bytes. \\n"
        "In the U-Frame, only one of the first six bits of the field \\\"ControlField1\\\" can be 1. \\n"
        "Protocol Specification:\\n"
        "=== BEGIN SPEC ===\\n"
        "%s\\n"
        "=== END SPEC ===\"}]",
        protocol_name, rfc_content) == -1) {
        free(rfc_content);
        return NULL;
    }

    free(rfc_content);
    return prompt;
}

// 2. Construct prompt for state machine analysis
char *construct_prompt_for_state_machine(const char *protocol_name, const char *spec_path) {
    char *prompt = NULL;
    char *rfc_content = NULL;
    
    // Read RFC content
    FILE *fp = fopen(spec_path, "r");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file %s\n", spec_path);
        return NULL;
    }
    
    // Get file size safely
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long file_size = ftell(fp);
    if (file_size == -1) {
        fclose(fp);
        return NULL;
    }
    rewind(fp);

    // Allocate memory for RFC content
    rfc_content = malloc(file_size + 1);
    if (rfc_content == NULL) {
        fprintf(stderr, "Error allocating memory for %s\n", spec_path);
        fclose(fp);
        return NULL;
    }

    // Read file content
    size_t bytes_read = fread(rfc_content, 1, file_size, fp);
    fclose(fp);
    if (bytes_read != (size_t)file_size) {
        free(rfc_content);
        return NULL;
    }
    rfc_content[file_size] = '\0';

    // Construct the prompt
    if (asprintf(&prompt, 
        "[{\"role\": \"user\", \"content\": \"You are a senior protocol analysis expert and you want to construct protocol messages for protocol fuzzing test. For the %s protocol, analyze the following protocol specification and get the **request-side protocol state machine**. \\n"   
        "The protocol state machine refer to the formal rules that define:\\n"
        "1. State transitions and actions\\n"
        "2. Conditions for state transitions\\n"
        "3. Events that trigger state transitions\\n\\n"
        "The protocol state machine should be organized in the Mermaid format. \\n"
        "For example, the request-side protocol state machine of the Modbus protocol is as follows: \\n"
        "stateDiagram-v2\\n"
        "    [*] --> IDLE\\n"
        "    IDLE --> CONNECTED : Connect\\n"
        "    CONNECTED --> WAITING_FOR_RESPONSE : SendRequest\\n"
        "    CONNECTED --> DISCONNECTED : Timeout\\n"
        "    WAITING_FOR_RESPONSE --> RESPONSE_RECEIVED : ReceiveResponse\\n"
        "    WAITING_FOR_RESPONSE --> DISCONNECTED : Timeout\\n"
        "    RESPONSE_RECEIVED --> [*]\\n"
        "    DISCONNECTED --> [*]\\n"
        "Protocol Specification:\\n"
        "=== BEGIN SPEC ===\\n"
        "%s\\n"
        "=== END SPEC ===\"}]",
        protocol_name, rfc_content) == -1) {
        free(rfc_content);
        return NULL;
    }

    free(rfc_content);
    return prompt;
}

// 3. Parse and save message grammar
void parse_and_save_message_grammar(const char *llm_response, const char *protocol_name, const char *output_dir) {
    if (!llm_response || !protocol_name || !output_dir) {
        FATAL("Invalid arguments to parse_and_save_message_grammar");
        return;
    }

    // Create output directory if it doesn't exist
    struct stat st = {0};
    if (stat(output_dir, &st) == -1) {
        mkdir(output_dir, 0777);
    }

    // Save JSON part
    char *json_filename = NULL;
    asprintf(&json_filename, "%s/%s_message.json", output_dir, protocol_name);
    if (!json_filename) {
        FATAL("Failed to allocate memory for JSON filename");
        return;
    }

    // Find JSON content
    const char *json_start = strchr(llm_response, '{');
    const char *json_end = strrchr(llm_response, '}');
    
    if (!json_start || !json_end) {
        FATAL("No valid JSON content found in LLM response");
        free(json_filename);
        return;
    }

    // Write JSON file
    FILE *json_fp = fopen(json_filename, "w");
    if (!json_fp) {
        FATAL("Failed to open JSON file for writing");
        free(json_filename);
        return;
    }

    size_t json_len = json_end - json_start + 1;
    fwrite(json_start, 1, json_len, json_fp);
    fclose(json_fp);
    free(json_filename);

    // Save text part
    char *txt_filename = NULL;
    asprintf(&txt_filename, "%s/%s_message.txt", output_dir, protocol_name);
    if (!txt_filename) {
        FATAL("Failed to allocate memory for text filename");
        return;
    }

    FILE *txt_fp = fopen(txt_filename, "w");
    if (!txt_fp) {
        FATAL("Failed to open text file for writing");
        free(txt_filename);
        return;
    }

    // Write text content (everything after the JSON)
    const char *text_content = json_end + 1;
    while (*text_content && (*text_content == '\n' || *text_content == ' ')) {
        text_content++;
    }
    if (*text_content) {
        fprintf(txt_fp, "%s", text_content);
    }

    fclose(txt_fp);
    free(txt_filename);
    
    printf("Saved message grammar to %s/%s_message.json and %s/%s_message.txt\n", 
           output_dir, protocol_name, output_dir, protocol_name);
}

// 4. Save state machine
void save_state_machine(const char *llm_response, const char *protocol_name, const char *output_dir) {
    if (!llm_response || !protocol_name || !output_dir) {
        FATAL("Invalid arguments to save_state_machine");
        return;
    }

    // Create output directory if it doesn't exist
    struct stat st = {0};
    if (stat(output_dir, &st) == -1) {
        mkdir(output_dir, 0777);
    }

    // Save Mermaid file
    char *mmd_filename = NULL;
    asprintf(&mmd_filename, "%s/%s_fsm.mmd", output_dir, protocol_name);
    if (!mmd_filename) {
        FATAL("Failed to allocate memory for Mermaid filename");
        return;
    }

    FILE *mmd_fp = fopen(mmd_filename, "w");
    if (!mmd_fp) {
        FATAL("Failed to open Mermaid file for writing");
        free(mmd_filename);
        return;
    }

    // Find Mermaid content
    const char *mmd_start = strstr(llm_response, "stateDiagram");
    if (!mmd_start) {
        FATAL("No valid Mermaid content found in LLM response");
        fclose(mmd_fp);
        free(mmd_filename);
        return;
    }

    // Write Mermaid file
    fprintf(mmd_fp, "%s", mmd_start);
    fclose(mmd_fp);
    free(mmd_filename);
    
    printf("Saved state machine to %s/%s_fsm.mmd\n", output_dir, protocol_name);
}

// Helper function to read a file into memory
static char* read_file_content(const char* filepath, bool* success) {
    FILE* fp = fopen(filepath, "r");
    if (!fp) {
        *success = false;
        return NULL;
    }
    
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    rewind(fp);
    
    char* content = malloc(file_size + 1);
    if (!content) {
        fclose(fp);
        *success = false;
        return NULL;
    }
    
    size_t bytes_read = fread(content, 1, file_size, fp);
    fclose(fp);
    
    if (bytes_read != (size_t)file_size) {
        free(content);
        *success = false;
        return NULL;
    }
    
    content[file_size] = '\0';
    *success = true;
    return content;
}

// Helper function to verify the JSON and TXT grammar file
static char* prompt_verify_json_txt_grammar(const char* protocol_name, const char* json_content, const char* txt_content, const char* rfc_content) {
    
    // Construct JSON and TXT grammar verification prompt
    char* verification_prompt = NULL;
    asprintf(&verification_prompt, 
        "[{\"role\": \"user\", \"content\": \"You are a senior industrial protocol analysis expert. You are given two files that together describe the request message grammar of industrial protocol %s: \\n"
        "1. A **JSON** file that defines the structured fields (field names, types, lengths, default values, etc.)\\n"
        "2. A **TXT** file that provides the additional constraints that are difficult to represent in JSON (e.g., field dependencies, the value range of the field, some fields can only have specific values, etc.)\\n"
        "Your task is to review both files **together** and determine whether the request message grammar is **complete, accurate, and self-consistent**.\\n"
        "Specifically, you should check the following: \\n"
        "1. Are there any required fields missing from the JSON? \\n"
        "2. Are all field types, lengths, and default values in the JSON correct? \\n"
        "3. Are all constraints described in the TXT file consistent with the JSON field structure? \\n"
        "4. Are any field dependencies, conditions, or value constraints **missing entirely** from both files? \\n"
        "5. Is the overall field order, type, and structure correct for the request message according to standard protocol layout? \\n"
        "If everything is correct, reply with: **true** as a JSON object: \\n"
        "{\\n"
        "  \\\"is_valid\\\": true,\\n"
        "}\\n"
        "If any issues are found, please output them as a JSON array of structured objects: \\n"
        "{\\n"
        "  \\\"is_valid\\\": false,\\n"
        "  \\\"issues\\\": [\\n"
        "    {\\n"
        "      \\\"issue_type\\\": \\\"missing_field\\\","
        "      \\\"description\\\": \\\"The field 'field_name' is missing from the JSON file\\\","
        "      \\\"affected_field\\\": \\\"field_name\\\","
        "      \\\"suggested_fix\\\": \\\"Add the field 'field_name' to the JSON file\\\"\\n"
        "    }\\n"
        "  ]\\n"
        "}\\n"
        "PROTOCOL SPECIFICATION:\\n"
        "=== BEGIN SPEC ===\\n"
        "%s"
        "=== END SPEC ===\\n\\n"
        "MESSAGE GRAMMAR (JSON):\\n"
        "=== BEGIN GRAMMAR (JSON) ===\\n"
        "%s"
        "=== END GRAMMAR (JSON) ===\\n\\n"
        "ADDITIONAL INFORMATION (TXT):\\n"
        "=== BEGIN ADDITIONAL INFORMATION (TXT) ===\\n"
        "%s"
        "=== END ADDITIONAL INFORMATION (TXT) ===\\n\\n\"}]",
        protocol_name,
        rfc_content,
        json_content,
        txt_content);
    
    if (!verification_prompt) {
        FATAL("Failed to construct JSON verification prompt");
        return NULL;
    }
    
    return verification_prompt;  
}

// Helper function to update the JSON and TXT additional info file
static char* prompt_update_json_txt_info(const char* protocol_name, const char* txt_content, const char* json_content, const char* rfc_content, const char* issues) {
    
    // Construct JSON and TXT verification prompt
    char* update_prompt = NULL;
    asprintf(&update_prompt, 
        "[{\"role\": \"user\", \"content\": \"You are a senior protocol parsing expert. You are given four inputs: \\n"
        "1. The original **JSON** string, which describes the structed request message structure of the %s industrial protocol. \\n"
        "2. The original **TXT** string, which includes additional protocol rules that cannot be easily represented in JSON (e.g., field dependencies, value constraints). \\n"
        "3. A JSON object structured **issue list**, where each entry describes a problem found in the original JSON or TXT file, along with a suggested fix. \\n"
        "4. The **protocol specification**, writen in natural language, which is the authoritative source describing the protocol's correct structure and behavior. \\n"
        "Your task is to generate a new versions of both the JSON and TXT files that resolve **all issues** listed by referring to the above four inputs: \\n"
        "- Resolve **allissues** in the issue list;\\n"
        "- Conform to the protocol specification (you may supplement the issue list with additional fixes if the spec reveals missing or incorrect details);\\n"
        "- Preserve the formatting style of the originals, the request message structure is orgized in the JSON and the additional information is orgized in the TXT;\\n"
        "You should **not attempt to modify the original inputs in place**. Instead, generate complete, corrected versions of both the JSON and the TXT from scratch.\\n"
        "### Output formatting requirements: \\n"
        "- The `updated_json` field must contain the **entire corrected JSON string**, preserving the original structure and style unless the issue list or specification requires changes. \\n"
        "- The `updated_txt` field must contain the **entire corrected TXT string**, preserving the original format (e.g., bullet lists, rule numbers, plain English). \\n"
        "- The returned JSON object must include **only** the two fields: `updated_json` and `updated_txt`. Do not include any commentary or explanations outside this structure. \\n"
        "- The two outputs must be suitable for **direct saving to files** and replacing the originals. \\n"
        "### Output format (required): \\n"
        "```json \\n"
        "{ \\n"
        "  \\\"updated_json\\\": \\\"<the complete corrected JSON string>\\\","
        "  \\\"updated_txt\\\": \\\"<the complete corrected TXT string>\\\" \\n"
        "}\\n"
        "### Here is the input content: \\n"
        "The original JSON:\\n"
        "=== BEGIN ORIGINAL JSON ===\\n"
        "%s"
        "=== END ORIGINAL JSON ===\\n\\n"
        "The original TXT:\\n"
        "=== BEGIN ORIGINAL TXT ===\\n"
        "%s"
        "=== END ORIGINAL TXT ===\\n\\n"
        "The issue list:\\n"
        "=== BEGIN ISSUE LIST ===\\n"
        "%s"
        "=== END ISSUE LIST ===\\n\\n"
        "PROTOCOL SPECIFICATION:\\n"
        "=== BEGIN SPEC ===\\n"
        "%s"
        "=== END SPEC ===\\n\\n\"}]",
        protocol_name,
        json_content,
        txt_content,
        issues,
        rfc_content);

    if (!update_prompt) {
        FATAL("Failed to construct JSON and TXT verification prompt");
        return NULL;
    }

    return update_prompt; 
}

// 5. Verify and update protocol grammar, additional info
bool verify_and_update_protocol_grammar(const char *protocol_name, const char *output_dir, const char *spec_path) {
    if (!protocol_name || !output_dir || !spec_path) {
        FATAL("Invalid arguments to verify_and_update_protocol_grammar");
        return false;
    }

    // Construct file paths for JSON and TXT files
    char *json_filename = NULL;
    char *txt_filename = NULL;
    asprintf(&json_filename, "%s/%s_message.json", output_dir, protocol_name);
    asprintf(&txt_filename, "%s/%s_message.txt", output_dir, protocol_name);
    
    if (!json_filename || !txt_filename) {
        FATAL("Failed to allocate memory for filenames");
        free(json_filename);
        free(txt_filename);
        return false;
    }

    // Read existing JSON and TXT content
    bool success;
    char *json_content = read_file_content(json_filename, &success);
    if (!success) {
        FATAL("Failed to read JSON file: %s", json_filename);
        free(json_filename);
        free(txt_filename);
        return false;
    }

    char *txt_content = read_file_content(txt_filename, &success);
    if (!success) {
        FATAL("Failed to read TXT file: %s", txt_filename);
        free(json_filename);
        free(txt_filename);
        free(json_content);
        return false;
    }

    // Read RFC content
    char *rfc_content = read_file_content(spec_path, &success);
    if (!success) {
        FATAL("Failed to read RFC file: %s", spec_path);
        free(json_filename);
        free(txt_filename);
        free(json_content);
        free(txt_content);
        return false;
    }

    // Generate verification prompt
    char *verification_prompt = prompt_verify_json_txt_grammar(protocol_name, json_content, txt_content, rfc_content);
    if (!verification_prompt) {
        FATAL("Failed to construct verification prompt");
        free(json_filename);
        free(txt_filename);
        free(json_content);
        free(txt_content);
        free(rfc_content);
        return false;
    }

    // Send verification prompt to LLM
    printf("Verifying protocol grammar for %s...\n", protocol_name);
    char *verification_response = chat_with_llm(verification_prompt, 3, 0.2);
    free(verification_prompt);
    
    if (!verification_response) {
        FATAL("Failed to get verification response from LLM");
        free(json_filename);
        free(txt_filename);
        free(json_content);
        free(txt_content);
        free(rfc_content);
        return false;
    }

    // Parse verification response to check if grammar is valid
    bool is_valid = false;
    const char *resp_start = strchr(verification_response, '{');
    const char *resp_end = strrchr(verification_response, '}');
    
    if (resp_start && resp_end) {
        size_t resp_len = resp_end - resp_start + 1;
        char *resp_str = malloc(resp_len + 1);
        if (resp_str) {
            memcpy(resp_str, resp_start, resp_len);
            resp_str[resp_len] = '\0';
            
            json_object *resp_obj = json_tokener_parse(resp_str);
            free(resp_str);
            
            if (resp_obj) {
                json_object *is_valid_obj;
                if (json_object_object_get_ex(resp_obj, "is_valid", &is_valid_obj)) {
                    is_valid = json_object_get_boolean(is_valid_obj);
                }
                
                if (is_valid) {
                    printf("Protocol grammar for %s is valid.\n", protocol_name);
                    json_object_put(resp_obj);
                    free(verification_response);
                    free(json_filename);
                    free(txt_filename);
                    free(json_content);
                    free(txt_content);
                    free(rfc_content);
                    return true;
                } else {
                    printf("Protocol grammar for %s has issues. Updating...\n", protocol_name);
                    
                    // Extract issues for update prompt
                    const char *issues_str = json_object_to_json_string(resp_obj);
                    
                    // Generate update prompt
                    char *update_prompt = prompt_update_json_txt_info(protocol_name, txt_content, json_content, rfc_content, issues_str);
                    if (!update_prompt) {
                        FATAL("Failed to construct update prompt");
                        json_object_put(resp_obj);
                        free(verification_response);
                        free(json_filename);
                        free(txt_filename);
                        free(json_content);
                        free(txt_content);
                        free(rfc_content);
                        return false;
                    }
                    
                    // Send update prompt to LLM
                    printf("Sending update request to LLM...\n");
                    char *update_response = chat_with_llm(update_prompt, 3, 0.5);
                    free(update_prompt);
                    
                    if (!update_response) {
                        FATAL("Failed to get update response from LLM");
                        json_object_put(resp_obj);
                        free(verification_response);
                        free(json_filename);
                        free(txt_filename);
                        free(json_content);
                        free(txt_content);
                        free(rfc_content);
                        return false;
                    }
                    
                    // Parse update response and save new files
                    const char *update_start = strchr(update_response, '{');
                    const char *update_end = strrchr(update_response, '}');
                    
                    if (update_start && update_end) {
                        size_t update_len = update_end - update_start + 1;
                        char *update_str = malloc(update_len + 1);
                        if (update_str) {
                            memcpy(update_str, update_start, update_len);
                            update_str[update_len] = '\0';
                            
                            json_object *update_obj = json_tokener_parse(update_str);
                            free(update_str);
                            
                            if (update_obj) {
                                json_object *updated_json_obj, *updated_txt_obj;
                                if (json_object_object_get_ex(update_obj, "updated_json", &updated_json_obj) &&
                                    json_object_object_get_ex(update_obj, "updated_txt", &updated_txt_obj)) {
                                    
                                    const char *new_json = json_object_get_string(updated_json_obj);
                                    const char *new_txt = json_object_get_string(updated_txt_obj);
                                    
                                    // Construct new filenames with "new_" prefix
                                    char *new_json_filename = NULL;
                                    char *new_txt_filename = NULL;
                                    asprintf(&new_json_filename, "%s/new_%s_message.json", output_dir, protocol_name);
                                    asprintf(&new_txt_filename, "%s/new_%s_message.txt", output_dir, protocol_name);
                                    
                                    if (!new_json_filename || !new_txt_filename) {
                                        FATAL("Failed to allocate memory for new filenames");
                                        free(new_json_filename);
                                        free(new_txt_filename);
                                        json_object_put(update_obj);
                                        free(update_response);
                                        json_object_put(resp_obj);
                                        free(verification_response);
                                        free(json_filename);
                                        free(txt_filename);
                                        free(json_content);
                                        free(txt_content);
                                        free(rfc_content);
                                        return false;
                                    }
                                    
                                    // Save updated JSON file
                                    FILE *json_fp = fopen(new_json_filename, "w");
                                    if (json_fp) {
                                        fprintf(json_fp, "%s", new_json);
                                        fclose(json_fp);
                                        printf("Updated JSON file saved: %s\n", new_json_filename);
                                    } else {
                                        FATAL("Failed to write updated JSON file: %s", new_json_filename);
                                    }
                                    
                                    // Save updated TXT file
                                    FILE *txt_fp = fopen(new_txt_filename, "w");
                                    if (txt_fp) {
                                        fprintf(txt_fp, "%s", new_txt);
                                        fclose(txt_fp);
                                        printf("Updated TXT file saved: %s\n", new_txt_filename);
                                    } else {
                                        FATAL("Failed to write updated TXT file: %s", new_txt_filename);
                                    }
                                    
                                    // Clean up new filename allocations
                                    free(new_json_filename);
                                    free(new_txt_filename);
                                } else {
                                    FATAL("Failed to extract updated_json or updated_txt from LLM response");
                                }
                                json_object_put(update_obj);
                            } else {
                                FATAL("Failed to parse update response JSON");
                            }
                        }
                    }
                    
                    free(update_response);
                }
                json_object_put(resp_obj);
            }
        }
    }
    
    // Cleanup
    free(verification_response);
    free(json_filename);
    free(txt_filename);
    free(json_content);
    free(txt_content);
    free(rfc_content);
    
    return !is_valid; // Return true if we successfully updated, false if original was valid
}

// Helper function to verify and update the MMD state machine file
static char* prompt_generate_state_machine_json(const char* protocol_name, const char* rfc_content) {
    // Construct state machine verification prompt
    char* state_machine_json_prompt = NULL;
    asprintf(&state_machine_json_prompt, 
        "[{\"role\": \"user\", \"content\": \"You are a senior industrial protocol analysis expert. You are given the specification of industrial protocol %s, please read the following protocol specification and extract the request-side protocol **state machine structure**, and output it in **JSON format**. \\n"
        "Your output must contain two parts: \\n"
        "1. A list of all unique **state names** used in the state machine; \\n"
        "2. A list of all **transitions (events)**, including: \\n"
        "   - The **event name** (or `null` if it is an implicit transition), \\n"
        "   - The **from-state**, \\n"
        "   - The **to-state**. \\n"
        "Please use the following JSON structure for your output: \\n"
        "```json \\n"
        "{ \\n"
        "  \\\"states\\\": [\\\"STATE1\\\", \\\"STATE2\\\", ...], \\n"
        "  \\\"events\\\": [ \\n"
        "    {\\\"event\\\": \\\"EventName\\\", \\\"from\\\": \\\"STATE_A\\\", \\\"to\\\": \\\"STATE_B\\\"}, \\n"
        "    {\\\"event\\\": null, \\\"from\\\": \\\"[*]\", \\\"to\\\": \\\"STATE_A\\\"}  // initial transition \\n"
        "  ] \\n"
        "}\\n"
        "``` \\n"
        "PROTOCOL SPECIFICATION:\\n"
        "=== BEGIN SPEC ===\\n"
        "%s"
        "=== END SPEC ===\\n\\n\"}]",
        protocol_name,
        rfc_content);
    
    if (!state_machine_json_prompt) {
        FATAL("Failed to construct JSON verification prompt");
        return NULL;
    }
    
    return state_machine_json_prompt;  
}

static char* prompt_verify_state_machine(const char* protocol_name, const char* mmd_content, const char* rfc_content, const char* json_state_machine) {
    // Construct state machine verification prompt
    char* verification_prompt = NULL;
    asprintf(&verification_prompt, 
        "[{\"role\": \"user\", \"content\": \"You are a senior industrial protocol analysis expert. You are given three files that together describe the state machine of industrial protocol %s: \\n"
        "1. A **Mermaid state diagram** representing the request-side state machine of an industrial protocol. \\n"
        "2. A **textual protocol specification** describing expected behaviors and transitions. \\n"
        "3. A **JSON file** describing the state and events of the state machine. \\n"
        "Your task is to evaluate whether the state machine is **complete and correct** based on the specification and the JSON file. \\n"
        "Please review the diagram for the following: \\n"
        "1. Is there an initial state defined using `[ * ] --> STATE`? \\n"
        "2. Is there a terminal state defined using `STATE --> [ * ]`? \\n"
        "3. Are all states reachable and connected? (No isolated or orphan states) \\n"
        "4. Are all important protocol events represented? (e.g., connect, send, receive, timeout, retry) \\n"
        "5. Are the transitions and flows logically consistent? (No dead ends unless terminal, no unreachable states) \\n"
        "6. Does the state machine match the expected request-side behavior described in the specification? \\n"
        "If everything is correct and complete, reply with: **true** as a JSON object: \\n"
        "{\\n"
        "  \\\"is_valid\\\": true,\\n"
        "}\\n"
        "If there are any issues, return a JSON array of structured problem reports. Each report should include: \\n"
        "{\\n"
        "  \\\"is_valid\\\": false,\\n"
        "  \\\"issues\\\": [\\n"
        "    {\\n"
        "       \\\"issue_type\\\": \\\"missing_state\\\",\\n"
        "       \\\"description\\\": \\\"The state machine is missing a state that is required by the specification.\\\",\\n"
        "       \\\"suggested_fix\\\": \\\"Add a new state called 'CONNECTING' to the state machine.\\\"\\n"
        "    }\\n"
        "  ]\\n"
        "}\\n"
        "Here is the protocol specification: \\n"
        "=== BEGIN SPECIFICATION ===\\n"
        "%s"
        "=== END SPECIFICATION ===\\n\\n"
        "Here is the state machine (Mermaid):\\n"
        "=== BEGIN STATE MACHINE(MERMAID) ===\\n"
        "%s"
        "=== END STATE MACHINE(MERMAID) ===\\n\\n"
        "Here is the state machine (JSON):\\n"
        "=== BEGIN STATE MACHINE(JSON) ===\\n"
        "%s"
        "=== END STATE MACHINE(JSON) ===\\n\\n\"}]",
        protocol_name,
        rfc_content,
        mmd_content,
        json_state_machine);
    
    if (!verification_prompt) {
        FATAL("Failed to construct state machine verification prompt");
        return NULL;
    }
    
    return verification_prompt;  
}

static char* prompt_update_state_machine(const char* protocol_name, const char* mmd_content, const char* rfc_content, const char* issues_state_machine, const char* json_state_machine) {
    // Construct TXT verification prompt
    char* update_prompt = NULL;
    asprintf(&update_prompt, 
        "[{\"role\": \"user\", \"content\": \"You are a %s protocol modeling expert. You are given the following inputs related to a protocol's request-side state machine: \\n"
        "1. A Mermaid **state diagram** (in `stateDiagram-v2` syntax), which represents the initial version of the protocol's request-side state machine. \\n"
        "2. A structured **issue list** (in JSON), which identifies specific problems found in the state diagram (such as missing states, missing transitions, incorrect logic, semantic gaps, etc.). \\n"
        "3. A JSON-formatted list of **states and events** extracted from the specification, which serves as a structural reference. It contains: \\n"
        "   - `states`: a list of states that should appear; \\n"
        "   - `events`: a list of transitions with `event`, `from`, and `to` fields. \\n"
        "4. The original **protocol specification**, which describes the expected behavior of the protocol in natural language. \\n"
        "Your task: \\n"
        "1. Generate a **new, corrected Mermaid state diagram** in `stateDiagram-v2` syntax. \\n"
        "2. Fix **all issues** reported in the issue list. \\n"
        "3. Use the `states` and `events` JSON as a reference to fill in any missing structure. \\n"
        "4. Carefully align the updated state diagram with the **protocol specification** to ensure semantic correctness. \\n"
        "5. If you detect any **additional issues** not listed in the issue file (e.g. missing transitions, ambiguous state behavior, naming inconsistencies), you must also fix or clarify them. \\n"
        "6. Include **initial state (`[*]`) and terminal state** where appropriate. \\n"
        "7. Maintain a clean and readable structure. \\n"
        "### Output format: \\n"
        "Only output the corrected Mermaid diagram in the following format: \\n"
        "```mermaid \\n"
        "stateDiagram-v2 \\n"
        "    [*] --> STATE_A \\n"
        "    STATE_A --> STATE_B : EventName \\n"
        "    STATE_B --> [*] \\n"
        "``` \\n"
        "Here is the input content: \\n"
        "The original Mermaid diagram:\\n"
        "=== BEGIN ORIGINAL MERMAID ===\\n"
        "%s"
        "=== END ORIGINAL MERMAID ===\\n\\n"
        "The issue list:\\n"
        "=== BEGIN ISSUE LIST ===\\n"
        "%s"
        "=== END ISSUE LIST ===\\n\\n"
        "The JSON-formatted list of states and events:\\n"
        "=== BEGIN JSON-FORMATTED LIST OF STATES AND EVENTS ===\\n"
        "%s"
        "=== END JSON-FORMATTED LIST OF STATES AND EVENTS ===\\n\\n"
        "PROTOCOL SPECIFICATION:\\n"
        "=== BEGIN SPEC ===\\n"
        "%s"
        "=== END SPEC ===\\n\\n\"}]",
        protocol_name,
        mmd_content,
        issues_state_machine,
        json_state_machine,
        rfc_content);

    if (!update_prompt) {
        FATAL("Failed to construct state machine update prompt");
        return NULL;
    }

    return update_prompt;   

    
}

// 6. Verify and update protocol state machine
bool verify_and_update_protocol_state_machine(const char *protocol_name, const char *output_dir, const char *spec_path) {
    if (!protocol_name || !output_dir || !spec_path) {
        FATAL("Invalid arguments to verify_and_update_protocol_state_machine");
        return false;
    }

    // Construct file path for existing MMD file
    char *mmd_filename = NULL;
    asprintf(&mmd_filename, "%s/%s_fsm.mmd", output_dir, protocol_name);
    if (!mmd_filename) {
        FATAL("Failed to allocate memory for MMD filename");
        return false;
    }

    // Read existing MMD content
    bool success;
    char *mmd_content = read_file_content(mmd_filename, &success);
    if (!success) {
        FATAL("Failed to read MMD file: %s", mmd_filename);
        free(mmd_filename);
        return false;
    }

    // Read RFC content
    char *rfc_content = read_file_content(spec_path, &success);
    if (!success) {
        FATAL("Failed to read RFC file: %s", spec_path);
        free(mmd_filename);
        free(mmd_content);
        return false;
    }

    // Step 1: Generate JSON state machine
    char *json_prompt = prompt_generate_state_machine_json(protocol_name, rfc_content);
    if (!json_prompt) {
        FATAL("Failed to construct JSON state machine prompt");
        free(mmd_filename);
        free(mmd_content);
        free(rfc_content);
        return false;
    }

    printf("Generating JSON state machine for %s...\n", protocol_name);
    char *json_response = chat_with_llm(json_prompt, 3, 0.2);
    free(json_prompt);

    if (!json_response) {
        FATAL("Failed to get JSON state machine response from LLM");
        free(mmd_filename);
        free(mmd_content);
        free(rfc_content);
        return false;
    }

    // Extract JSON from response
    const char *json_start = strchr(json_response, '{');
    const char *json_end = strrchr(json_response, '}');
    char *json_state_machine = NULL;
    
    if (json_start && json_end) {
        size_t json_len = json_end - json_start + 1;
        json_state_machine = malloc(json_len + 1);
        if (json_state_machine) {
            memcpy(json_state_machine, json_start, json_len);
            json_state_machine[json_len] = '\0';
        }
    }

    if (!json_state_machine) {
        FATAL("Failed to extract JSON state machine from LLM response");
        free(json_response);
        free(mmd_filename);
        free(mmd_content);
        free(rfc_content);
        return false;
    }

    // Step 2: Verify state machine
    char *verification_prompt = prompt_verify_state_machine(protocol_name, mmd_content, rfc_content, json_state_machine);
    if (!verification_prompt) {
        FATAL("Failed to construct verification prompt");
        free(json_response);
        free(json_state_machine);
        free(mmd_filename);
        free(mmd_content);
        free(rfc_content);
        return false;
    }

    printf("Verifying state machine for %s...\n", protocol_name);
    char *verification_response = chat_with_llm(verification_prompt, 3, 0.2);
    free(verification_prompt);

    if (!verification_response) {
        FATAL("Failed to get verification response from LLM");
        free(json_response);
        free(json_state_machine);
        free(mmd_filename);
        free(mmd_content);
        free(rfc_content);
        return false;
    }

    // Parse verification response
    bool is_valid = false;
    const char *resp_start = strchr(verification_response, '{');
    const char *resp_end = strrchr(verification_response, '}');
    
    if (resp_start && resp_end) {
        size_t resp_len = resp_end - resp_start + 1;
        char *resp_str = malloc(resp_len + 1);
        if (resp_str) {
            memcpy(resp_str, resp_start, resp_len);
            resp_str[resp_len] = '\0';
            
            json_object *resp_obj = json_tokener_parse(resp_str);
            free(resp_str);
            
            if (resp_obj) {
                json_object *is_valid_obj;
                if (json_object_object_get_ex(resp_obj, "is_valid", &is_valid_obj)) {
                    is_valid = json_object_get_boolean(is_valid_obj);
                }
                
                if (is_valid) {
                    printf("State machine for %s is valid.\n", protocol_name);
                    json_object_put(resp_obj);
                    free(verification_response);
                    free(json_response);
                    free(json_state_machine);
                    free(mmd_filename);
                    free(mmd_content);
                    free(rfc_content);
                    return true;
                } else {
                    printf("State machine for %s has issues. Updating...\n", protocol_name);
                    
                    // Extract issues for update prompt
                    const char *issues_str = json_object_to_json_string(resp_obj);
                    
                    // Step 3: Update state machine
                    char *update_prompt = prompt_update_state_machine(protocol_name, mmd_content, rfc_content, issues_str, json_state_machine);
                    if (!update_prompt) {
                        FATAL("Failed to construct update prompt");
                        json_object_put(resp_obj);
                        free(verification_response);
                        free(json_response);
                        free(json_state_machine);
                        free(mmd_filename);
                        free(mmd_content);
                        free(rfc_content);
                        return false;
                    }
                    
                    printf("Sending update request to LLM...\n");
                    char *update_response = chat_with_llm(update_prompt, 3, 0.5);
                    free(update_prompt);
                    
                    if (!update_response) {
                        FATAL("Failed to get update response from LLM");
                        json_object_put(resp_obj);
                        free(verification_response);
                        free(json_response);
                        free(json_state_machine);
                        free(mmd_filename);
                        free(mmd_content);
                        free(rfc_content);
                        return false;
                    }
                    
                    // Extract and save new Mermaid diagram
                    const char *mermaid_start = strstr(update_response, "stateDiagram");
                    if (mermaid_start) {
                        // Save new MMD file with "new_" prefix
                        char *new_mmd_filename = NULL;
                        asprintf(&new_mmd_filename, "%s/new_%s_fsm.mmd", output_dir, protocol_name);
                        
                        if (new_mmd_filename) {
                            FILE *mmd_fp = fopen(new_mmd_filename, "w");
                            if (mmd_fp) {
                                fprintf(mmd_fp, "%s", mermaid_start);
                                fclose(mmd_fp);
                                printf("Updated state machine saved: %s\n", new_mmd_filename);
                            } else {
                                FATAL("Failed to write updated MMD file: %s", new_mmd_filename);
                            }
                            free(new_mmd_filename);
                        }
                    } else {
                        FATAL("Failed to extract Mermaid diagram from update response");
                    }
                    
                    free(update_response);
                }
                json_object_put(resp_obj);
            }
        }
    }

    // Cleanup
    free(verification_response);
    free(json_response);
    free(json_state_machine);
    free(mmd_filename);
    free(mmd_content);
    free(rfc_content);
    
    return !is_valid; // Return true if we successfully updated, false if original was valid
}

/*bool protocol_analysis_and_update(const char *protocol_name, const char *spec_path, const char *output_dir) {

    printf("Protocol Analysis\n");
    printf("=================================\n");
    printf("Protocol: %s\n", protocol_name);
    printf("Spec file: %s\n", spec_path);
    printf("Output directory: %s\n\n", output_dir);

    // Phase 1: Initial Protocol Analysis (same as TEST_PROTOCOL_ANALYSIS)
    printf("=== PHASE 1: INITIAL PROTOCOL ANALYSIS ===\n\n");

    // Step 1: Generate protocol message grammar
    printf("Step 1: Generating protocol message grammar...\n");
    char *grammar_prompt = construct_prompt_for_message_grammar(protocol_name, spec_path);
    if (!grammar_prompt) {
        fprintf(stderr, "Error: Failed to construct grammar prompt\n");
        return false;
    }

    printf("Grammar prompt constructed successfully\n");
    //printf("Sending request to OpenAI API...\n");
    
    char *grammar_response = chat_with_llm(grammar_prompt, 3, 0.5);
    free(grammar_prompt);
    
    if (!grammar_response) {
        fprintf(stderr, "Error: Failed to get grammar response from LLM\n");
        return false;
    }

    //printf("LLM response received successfully\n");
    printf("Parsing and saving message grammar...\n");
    
    parse_and_save_message_grammar(grammar_response, protocol_name, output_dir);
    free(grammar_response);
    
    printf("Message grammar saved successfully\n\n");

    // Step 2: Generate protocol state machine
    printf("Step 2: Generating protocol state machine...\n");
    char *fsm_prompt = construct_prompt_for_state_machine(protocol_name, spec_path);
    if (!fsm_prompt) {
        fprintf(stderr, "Error: Failed to construct state machine prompt\n");
        return false;
    }

    printf("State machine prompt constructed successfully\n");
    //printf("Sending request to OpenAI API...\n");
    
    char *fsm_response = chat_with_llm(fsm_prompt, 3, 0.5);
    free(fsm_prompt);
    
    if (!fsm_response) {
        fprintf(stderr, "Error: Failed to get state machine response from LLM\n");
        return false;
    }

    //printf("LLM response received successfully\n");
    //printf("Saving state machine...\n");
    
    save_state_machine(fsm_response, protocol_name, output_dir);
    free(fsm_response);
    
    printf("State machine saved successfully\n\n");

    // Phase 2: Verification and Update
    printf("=== PHASE 2: VERIFICATION AND IMPROVEMENT ===\n\n");

    // Step 3: Verify and update protocol grammar (JSON + TXT)
    printf("Step 3: Verifying and updating protocol grammar...\n");
    bool grammar_updated = verify_and_update_protocol_grammar(protocol_name, output_dir, spec_path);
    
    if (grammar_updated) {
        printf("Protocol grammar was updated successfully!\n");
        printf("   - Check new_%s_message.json for updated JSON grammar\n", protocol_name);
        printf("   - Check new_%s_message.txt for updated additional information\n", protocol_name);
    } else {
        printf("Protocol grammar verification passed - no updates needed\n");
    }
    printf("\n");

    // Step 4: Verify and update protocol state machine (MMD)
    printf("Step 4: Verifying and updating protocol state machine...\n");
    bool state_machine_updated = verify_and_update_protocol_state_machine(protocol_name, output_dir, spec_path);
    
    if (state_machine_updated) {
        printf("Protocol state machine was updated successfully!\n");
        printf("   - Check new_%s_fsm.mmd for updated state machine\n", protocol_name);
    } else {
        printf("Protocol state machine verification passed - no updates needed\n");
    }
    printf("\n");

    // Summary
    printf("=== SUMMARY ===\n");
    printf("Protocol analysis and verification completed!\n\n");
    
    printf("Generated files:\n");
    printf("%s/%s_message.json: Protocol message grammar (JSON)\n", output_dir, protocol_name);
    printf("%s/%s_message.txt: Additional protocol constraints (TXT)\n", output_dir, protocol_name);
    printf("%s/%s_fsm.mmd: Protocol state machine (Mermaid)\n", output_dir, protocol_name);
    
    if (grammar_updated || state_machine_updated) {
        printf("\nImproved files:\n");
        if (grammar_updated) {
            printf("%s/new_%s_message.json: Updated protocol grammar (JSON)\n", output_dir, protocol_name);
            printf("%s/new_%s_message.txt: Updated additional information (TXT)\n", output_dir, protocol_name);
        }
        if (state_machine_updated) {
            printf("%s/new_%s_fsm.mmd: Updated state machine (Mermaid)\n", output_dir, protocol_name);
        }
        printf("\nThe 'new_' prefixed files contain the improved versions based on LLM verification.\n");
    }

    printf("\nProtocol analysis completed successfully!\n");
    
    return true;
}*/

//=============================following are the functions for bug detection (consistency analysis) ==========================================

// Global variables for RFC consistency analysis
static consistency_collect_mode_t rfc_collect_mode = COLLECT_DISABLED;
static unsigned int rfc_sampling_rate = 10;  // sampling rate (‰)
static unsigned char rfc_collection_active = 0;
static unsigned int global_interaction_id = 0;
static char *rfc_consistency_dir = NULL;
static char *rfc_interactions_dir = NULL;
static char *rfc_llm_analysis_dir = NULL;

/**
 * Initialize RFC consistency analysis system
 * @param output_dir: AFL output directory
 * @return: 0 on success, -1 on failure
 */
int init_rfc_consistency_analysis(const char *output_dir) {
    if (!output_dir) {
        printf("[-] ERROR: Output directory not specified for RFC consistency analysis\n");
        return -1;
    }
    
    // Create main consistency analysis directory
    rfc_consistency_dir = malloc(strlen(output_dir) + 20);  // "/rfc-consistency" + null terminator
    snprintf(rfc_consistency_dir, strlen(output_dir) + 20, "%s/rfc-consistency", output_dir);
    
    if (mkdir(rfc_consistency_dir, 0700) && errno != EEXIST) {
        printf("[-] ERROR: Unable to create RFC consistency directory '%s': %s\n", 
               rfc_consistency_dir, strerror(errno));
        free(rfc_consistency_dir);
        rfc_consistency_dir = NULL;
        return -1;
    }
    
    // Create interactions subdirectory
    rfc_interactions_dir = malloc(strlen(rfc_consistency_dir) + 15);  // "/interactions" + null terminator
    snprintf(rfc_interactions_dir, strlen(rfc_consistency_dir) + 15, "%s/interactions", rfc_consistency_dir);
    
    if (mkdir(rfc_interactions_dir, 0700) && errno != EEXIST) {
        printf("[-] ERROR: Unable to create interactions directory '%s': %s\n", 
               rfc_interactions_dir, strerror(errno));
        cleanup_rfc_consistency_analysis();
        return -1;
    }
    
    // Create LLM analysis results subdirectory
    rfc_llm_analysis_dir = malloc(strlen(rfc_consistency_dir) + 15);  // "/llm-analysis" + null terminator
    snprintf(rfc_llm_analysis_dir, strlen(rfc_consistency_dir) + 15, "%s/llm-analysis", rfc_consistency_dir);
    
    if (mkdir(rfc_llm_analysis_dir, 0700) && errno != EEXIST) {
        printf("[-] ERROR: Unable to create LLM analysis directory '%s': %s\n", 
               rfc_llm_analysis_dir, strerror(errno));
        cleanup_rfc_consistency_analysis();
        return -1;
    }
    
    // Initialize global state
    global_interaction_id = 0;
    rfc_collection_active = 1;
    
    printf("RFC Consistency Analysis Enabled\n");
    printf("=================================\n");
    printf("RFC consistency analysis initialized successfully!\n");
    printf("    - Main directory: %s\n", rfc_consistency_dir);
    printf("    - Interactions: %s\n", rfc_interactions_dir);
    printf("    - LLM analysis: %s\n", rfc_llm_analysis_dir);
    
    return 0;
}

/**
 * Cleanup RFC consistency analysis system
 */
void cleanup_rfc_consistency_analysis(void) {
    if (rfc_consistency_dir) {
        free(rfc_consistency_dir);
        rfc_consistency_dir = NULL;
    }
    if (rfc_interactions_dir) {
        free(rfc_interactions_dir);
        rfc_interactions_dir = NULL;
    }
    if (rfc_llm_analysis_dir) {
        free(rfc_llm_analysis_dir);
        rfc_llm_analysis_dir = NULL;
    }
    
    rfc_collection_active = 0;
    global_interaction_id = 0;
    
    printf("[*] RFC consistency analysis cleanup completed\n");
}

/**
 * Set RFC consistency data collection mode
 * @param mode: Collection mode (0=disabled, 1=new coverage only, 2=with sampling)
 * @param sampling_rate: Sampling rate in per mille (0-1000)
 */
void set_rfc_consistency_mode(consistency_collect_mode_t mode, unsigned int sampling_rate) {
    rfc_collect_mode = mode;
    if (sampling_rate <= 1000) {
        rfc_sampling_rate = sampling_rate;
    }
    
    printf("[*] RFC consistency mode set to %d, sampling rate: %u‰\n", mode, rfc_sampling_rate);
}

/**
 * Check if we should collect consistency data for this execution
 * @param has_new_bits_value: Value returned by has_new_bits() (0=no new bits, 1=new coverage, 2=new paths)
 * @return: 1 if should collect, 0 otherwise
 */
int should_collect_consistency_data(unsigned char has_new_bits_value) {
    if (!rfc_collection_active || rfc_collect_mode == COLLECT_DISABLED) {
        return 0;
    }
    
    switch (rfc_collect_mode) {
        case COLLECT_NEW_COVERAGE_ONLY:
            return (has_new_bits_value == 2);  // Only true new paths
            
        case COLLECT_WITH_SAMPLING:
            // Always collect new coverage or new paths
            if (has_new_bits_value == 2 || has_new_bits_value == 1) {
                return 1;
            }
            
            // Sample from executions with no new coverage
            if (has_new_bits_value == 0) {
                // Simple pseudo-random sampling
                static unsigned int sample_counter = 0;
                sample_counter++;
                return ((sample_counter * 1019) % 1000) < rfc_sampling_rate;
            }
            
            return 0;  // Should not reach here
            
        default:
            return 0;
    }
}

/**
 * Convert bytes to hexadecimal string
 * @param data: Input byte array
 * @param length: Length of input data
 * @return: Allocated hex string (caller must free), or NULL on error
 */
char *bytes_to_hex_string(const unsigned char *data, unsigned int length) {
    if (!data || length == 0) {
        char *empty = malloc(1);
        if (empty) empty[0] = '\0';
        return empty;
    }
    
    char *hex_string = malloc(length * 2 + 1);
    if (!hex_string) {
        printf("[-] ERROR: Memory allocation failed for hex string\n");
        return NULL;
    }
    
    for (unsigned int i = 0; i < length; i++) {
        sprintf(hex_string + i * 2, "%02x", data[i]);
    }
    hex_string[length * 2] = '\0';
    
    return hex_string;
}

/**
 * Get current timestamp as string
 * @return: Allocated timestamp string (caller must free)
 */
char *get_current_timestamp_string(void) {
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    
    char *timestamp = malloc(32);
    if (timestamp) {
        strftime(timestamp, 32, "%Y-%m-%d %H:%M:%S", tm_info);
    }
    
    return timestamp;
}

/**
 * Read entire file content into memory
 * @param file_path: Path to file
 * @return: Allocated file content (caller must free), or NULL on error
 */
char *read_entire_file(const char *file_path) {
    FILE *file = fopen(file_path, "r");
    if (!file) {
        printf("[-] ERROR: Cannot open file '%s': %s\n", file_path, strerror(errno));
        return NULL;
    }
    
    // Get file size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    if (file_size <= 0) {
        printf("[-] ERROR: Invalid file size for '%s'\n", file_path);
        fclose(file);
        return NULL;
    }
    
    // Allocate memory and read file
    char *content = malloc(file_size + 1);
    if (!content) {
        printf("[-] ERROR: Memory allocation failed for file content\n");
        fclose(file);
        return NULL;
    }
    
    size_t bytes_read = fread(content, 1, file_size, file);
    content[bytes_read] = '\0';
    
    fclose(file);
    return content;
}

/**
 * Collect interaction data from AFL's kl_messages and response buffers
 * @param kl_messages_ptr: Pointer to AFL's kl_messages linked list
 * @param response_buf_ptr: Pointer to AFL's response buffer
 * @param response_bytes_ptr: Pointer to AFL's response_bytes array
 * @param messages_sent_count: Number of messages sent
 * @return: 0 on success, -1 on failure
 */
int collect_interaction_data(void *kl_messages_ptr, void *response_buf_ptr, 
                           unsigned int *response_bytes_ptr, unsigned int messages_sent_count) {
    
    if (!rfc_collection_active || !rfc_interactions_dir) {
        return -1;
    }
    
    if (!kl_messages_ptr) {
        // Silently skip to avoid performance impact
        return -1;
    }
    
    if (messages_sent_count == 0) {
        // Silently skip to avoid performance impact
        return -1;
    }
    
    // Cast AFL data structures to proper types
    klist_t(lms) *kl_messages = (klist_t(lms) *)kl_messages_ptr;
    char *response_buf = (char *)response_buf_ptr;
    
    // Generate unique interaction ID
    unsigned int interaction_id = ++global_interaction_id;
    
    // Create interaction file path
    char *interaction_file = malloc(strlen(rfc_interactions_dir) + 30);  // "/interaction-XXXXXX.jsonl" + null
    if (!interaction_file) {
        printf("[-] ERROR: Memory allocation failed for interaction file path\n");
        return -1;
    }
    snprintf(interaction_file, strlen(rfc_interactions_dir) + 30, 
             "%s/interaction-%06u.jsonl", rfc_interactions_dir, interaction_id);
    
    FILE *file = fopen(interaction_file, "w");
    if (!file) {
        printf("[-] ERROR: Cannot create interaction file '%s': %s\n", 
               interaction_file, strerror(errno));
        free(interaction_file);
        return -1;
    }
    
    // Removed printf to avoid performance impact in main loop
    
    
    // Iterate through kl_messages linked list to extract request data
    kliter_t(lms) *iter = kl_begin(kl_messages);
    unsigned int seq = 1;
    unsigned int cumulative_response_len = 0;
    
    while (iter && iter != kl_end(kl_messages) && seq <= messages_sent_count) {
        message_t *msg = kl_val(iter);
        
        fprintf(file, "{\"sequence\":%u,", seq);
        
        // Extract request data and convert to hex
        if (msg && msg->mdata && msg->msize > 0) {
            char *request_hex = bytes_to_hex_string((unsigned char *)msg->mdata, msg->msize);
            fprintf(file, "\"request\":\"%s\",", request_hex ? request_hex : "");
            if (request_hex) free(request_hex);
        } else {
            fprintf(file, "\"request\":\"\",");
        }
        
        // Extract response data if available
        if (response_buf_ptr && response_bytes_ptr && seq <= messages_sent_count) {
            unsigned int response_start = cumulative_response_len;
            unsigned int response_end = response_bytes_ptr[seq-1];
            unsigned int response_len = response_end - response_start;
            
            if (response_len > 0 && response_buf && response_start < response_end) {
                // Convert response data to hex
                char *response_hex = bytes_to_hex_string(
                    (unsigned char *)(response_buf + response_start), response_len);
                fprintf(file, "\"response\":\"%s\"", response_hex ? response_hex : "");
                if (response_hex) free(response_hex);
            } else {
                fprintf(file, "\"response\":null");
            }
            
            cumulative_response_len = response_end;
        } else {
            fprintf(file, "\"response\":null");
        }
        
        fprintf(file, "}\n");
        
        // Move to next message
        iter = kl_next(iter);
        seq++;
    }
    
    // If we have fewer messages in kl_messages than messages_sent_count,
    // create placeholder entries for the remaining ones
    while (seq <= messages_sent_count) {
        fprintf(file, "{\"sequence\":%u,", seq);
        fprintf(file, "\"request\":\"unknown_message\",");
        
        if (response_buf_ptr && response_bytes_ptr && seq <= messages_sent_count) {
            unsigned int response_start = cumulative_response_len;
            unsigned int response_end = response_bytes_ptr[seq-1];
            unsigned int response_len = response_end - response_start;
            
            if (response_len > 0 && response_buf && response_start < response_end) {
                char *response_hex = bytes_to_hex_string(
                    (unsigned char *)(response_buf + response_start), response_len);
                fprintf(file, "\"response\":\"%s\"", response_hex ? response_hex : "");
                if (response_hex) free(response_hex);
            } else {
                fprintf(file, "\"response\":null");
            }
            
            cumulative_response_len = response_end;
        } else {
            fprintf(file, "\"response\":null");
        }
        
        fprintf(file, "}\n");
        seq++;
    }
    
    fclose(file);
    free(interaction_file);
    
    return 0;
}

/**
 * Construct LLM prompt for RFC consistency analysis
 * @param interaction_content: Content of the interaction JSONL file
 * @param rfc_content: Content of the RFC specification
 * @param interaction_id: ID of the interaction being analyzed
 * @return: Allocated prompt string (caller must free), or NULL on error
 */
static char *construct_rfc_analysis_prompt(const char *interaction_content, 
                                         const char *rfc_content, 
                                         const char *interaction_id,
                                         const char *protocol_name,
                                         const char *output_dir) {
    if (!interaction_content || !rfc_content || !interaction_id) {
        return NULL;
    }

    char *parse_result_path = NULL; 
    if (asprintf(&parse_result_path, "%s/%s-parse-result", output_dir, protocol_name) == -1) {
        printf("[-] ERROR: Memory allocation failed for parse result path\n");
        return NULL;
    }

    // Read protocol parse content (grammar and constraints)
    char *json_content = NULL;
    char *txt_content = NULL;
    char *state_machine_content = NULL;
    
    // Check for new_ versions first, then fall back to regular versions
    char *json_path = NULL;
    char *txt_path = NULL;
    char *state_machine_path = NULL;

    // Try new_ versions first
    if (asprintf(&json_path, "%s/new_%s_message.json", parse_result_path, protocol_name) == -1) {
        json_path = NULL;
    }
    if (asprintf(&txt_path, "%s/new_%s_message.txt", parse_result_path, protocol_name) == -1) {
        txt_path = NULL;
    }
    if (asprintf(&state_machine_path, "%s/new_%s_fsm.mmd", parse_result_path, protocol_name) == -1) {
        state_machine_path = NULL;
    }
    
    // Check if new_ versions exist, otherwise use regular versions
    if (json_path && access(json_path, F_OK) != 0) {
        free(json_path);
        if (asprintf(&json_path, "%s/%s_message.json", parse_result_path, protocol_name) == -1) {
            json_path = NULL;
        }
    }
    
    if (txt_path && access(txt_path, F_OK) != 0) {
        free(txt_path);
        if (asprintf(&txt_path, "%s/%s_message.txt", parse_result_path, protocol_name) == -1) {
            txt_path = NULL;
        }
    }

    if (state_machine_path && access(state_machine_path, F_OK) != 0) {
        free(state_machine_path);
        if (asprintf(&state_machine_path, "%s/%s_fsm.mmd", parse_result_path, protocol_name) == -1) {
            state_machine_path = NULL;
        }
    }
    
    // Read JSON grammar file
    if (json_path) {
        FILE *json_file = fopen(json_path, "r");
        if (json_file) {
            if (fseek(json_file, 0, SEEK_END) == 0) {
                long json_size = ftell(json_file);
                if (json_size > 0) {
                    rewind(json_file);
                    json_content = malloc(json_size + 1);
                    if (json_content && fread(json_content, 1, json_size, json_file) == (size_t)json_size) {
                        json_content[json_size] = '\0';
                    } else {
                        free(json_content);
                        json_content = NULL;
                    }
                }
            }
            fclose(json_file);
        }
        free(json_path);
    }
    
    // Read TXT constraints file
    if (txt_path) {
        FILE *txt_file = fopen(txt_path, "r");
        if (txt_file) {
            if (fseek(txt_file, 0, SEEK_END) == 0) {
                long txt_size = ftell(txt_file);
                if (txt_size > 0) {
                    rewind(txt_file);
                    txt_content = malloc(txt_size + 1);
                    if (txt_content && fread(txt_content, 1, txt_size, txt_file) == (size_t)txt_size) {
                        txt_content[txt_size] = '\0';
                    } else {
                        free(txt_content);
                        txt_content = NULL;
                    }
                }
            }
            fclose(txt_file);
        }
        free(txt_path);
    }

    // Read state machine file
    if (state_machine_path) {
        FILE *state_machine_file = fopen(state_machine_path, "r");
        if (state_machine_file) {
            if (fseek(state_machine_file, 0, SEEK_END) == 0) {
                long state_machine_size = ftell(state_machine_file);
                if (state_machine_size > 0) {
                    rewind(state_machine_file);
                    state_machine_content = malloc(state_machine_size + 1);
                    if (state_machine_content && fread(state_machine_content, 1, state_machine_size, state_machine_file) == (size_t)state_machine_size) {
                        state_machine_content[state_machine_size] = '\0';
                    } else {
                        free(state_machine_content);
                        state_machine_content = NULL;
                    }
                }
            }
            fclose(state_machine_file);
        }
        free(state_machine_path);
    }

    char *prompt = NULL;

    // Construct the prompt
    if (asprintf(&prompt, "[{\"role\": \"user\", \"content\": \"You are a protocol analysis expert. Your task is to evaluate whether a given interaction sequence between a client and server complies with the protocol's specification. You are given four inputs:\n\n"
    "1. **Protocol Specification** (natural language): This is the authoritative source describing the protocol structure, behavior, field encoding, message types, constraints, etc.\n"
    "2. **Protocol Grammar JSON**: This defines the message format, fields, types, order, and encoding rules.\n"
    "3. **Protocol Rules TXT**: This contains additional rules or constraints that are hard to represent in JSON, such as field dependencies, timing requirements, or value restrictions.\n"
    "4. **Protocol State Machine (Mermaid)**: This defines the client-side valid sequences of message exchanges, including states, transitions, and expected message types.\n"
    "5. **Interaction Sequence**: A chronological list of request and response messages exchanged between client and server, in hex-encoded format. Each step includes both a request and the corresponding response.\n\n"
    "### Your Evaluation Tasks:\n\n"
    "- **Step-by-step check** whether each request and response pair conforms to the protocol grammar and constraints.\n"
    "- **Validate state transitions** using the provided state machine. Ensure each request leads to a valid next state and is followed by an appropriate response.\n"
    "- **Report violations**, such as:\n"
    "  - Field format or length mismatches.\n"
    "  - Values violating constraints or field dependencies.\n"
    "  - Invalid message order or illegal transitions.\n"
    "  - Missing or incorrect responses.\n\n"
    "---\n\n"
    "### Output Format:\n\n"
    "Your output can be a clear, human-readable **plain text report** that is easy to understand and follow. \n\n"
    "### Provided Inputs:\n\n"
    "- Protocol Specification:\n"
    "=== BEGIN SPEC ===\n"
    "%s\n"
    "=== END SPEC ===\n\n"
    "- Protocol Grammar JSON:\n"
    "=== BEGIN JSON ===\n"
    "%s\n"
    "=== END JSON ===\n\n"
    "- Protocol Rules TXT:\n"
    "=== BEGIN TXT ===\n"
    "%s\n"
    "=== END TXT ===\n\n"
    "- Protocol State Machine:\n"
    "=== BEGIN MERMAID ===\n"
    "%s\n"
    "=== END MERMAID ===\n\n"
    "- Request/Response Sequence (Hex, Step-by-Step):\n"
    "=== BEGIN SEQUENCE ===\n"
    "%s\n"
    "=== END SEQUENCE ===\n\"}]",
    rfc_content, 
    json_content ? json_content : "", 
    txt_content ? txt_content : "", 
    state_machine_content ? state_machine_content : "",
    interaction_content) == -1) {
        FATAL("Failed to construct bug detection prompt");
        goto cleanup;
    }
    
    free(parse_result_path);  // Fix memory leak: free parse_result_path before returning
    return prompt;
    
    cleanup:
        // Cleanup
        free(parse_result_path);  // Fixed: added missing free for parse_result_path
        free(json_content);
        free(txt_content);
        free(state_machine_content);
        free(prompt);
        return NULL;
}

/**
 * Analyze a single interaction file using LLM
 * @param interaction_file_path: Path to the interaction JSONL file
 * @param interaction_id: ID string for the interaction
 * @param rfc_content: RFC specification content
 * @return: 0 on success, -1 on failure
 */
static int analyze_single_interaction_file(const char *interaction_file_path, 
                                         const char *interaction_id,
                                         const char *rfc_content,
                                        const char *protocol_name,
                                        const char *output_dir) {
    
    // Read interaction data
    char *interaction_content = read_entire_file(interaction_file_path);
    if (!interaction_content) {
        printf("[-] ERROR: Failed to read interaction file '%s'\n", interaction_file_path);
        return -1;
    }
    
    // Construct LLM prompt
    char *prompt = construct_rfc_analysis_prompt(interaction_content, rfc_content, interaction_id, protocol_name, output_dir);
    if (!prompt) {
        printf("[-] ERROR: Failed to construct analysis prompt for interaction %s\n", interaction_id);
        free(interaction_content);
        return -1;
    }
    
    printf("[*] Analyzing interaction %s with LLM...\n", interaction_id);
    
    // Call LLM for analysis
    char *llm_response = chat_with_llm(prompt, RFC_CONSISTENCY_RETRIES, 0.3);
    
    if (llm_response) {
        // Save analysis result
        char *result_file_path = malloc(strlen(rfc_llm_analysis_dir) + strlen(interaction_id) + 32);
        if (!result_file_path) {
            printf("[-] ERROR: Memory allocation failed for result file path\n");
            free(prompt);
            free(interaction_content);
            free(llm_response);
            return -1;
        }
        snprintf(result_file_path, strlen(rfc_llm_analysis_dir) + strlen(interaction_id) + 32,
                "%s/interaction-%s.txt", rfc_llm_analysis_dir, interaction_id);
        
        FILE *result_file = fopen(result_file_path, "w");
        if (result_file) {
            fprintf(result_file, "RFC Consistency Analysis Report\n");
            fprintf(result_file, "================================\n\n");
            fprintf(result_file, "Interaction ID: %s\n", interaction_id);
            fprintf(result_file, "Protocol: %s\n\n", protocol_name);  
            fprintf(result_file, "%s\n", llm_response);
            
            fclose(result_file);
            
            printf("[*] Analysis result saved to: %s\n", result_file_path);
        } else {
            printf("[-] ERROR: Failed to create result file '%s': %s\n", 
                   result_file_path, strerror(errno));
        }
        
        free(result_file_path);
        free(llm_response);
    } else {
        printf("[-] ERROR: LLM analysis failed for interaction %s\n", interaction_id);
    }
    
    free(prompt);
    free(interaction_content);
    
    return llm_response ? 0 : -1;
}

/**
 * Perform RFC consistency analysis on all collected interactions
 * @param rfc_path: Path to RFC specification file
 * @param protocol_name: Name of the protocol being analyzed
 * @return: Number of interactions analyzed, or -1 on failure
 */
int perform_rfc_consistency_analysis(const char *rfc_path, const char *protocol_name, const char *output_dir) {
    if (!rfc_collection_active || !rfc_interactions_dir || !rfc_llm_analysis_dir) {
        printf("[-] ERROR: RFC consistency analysis not properly initialized\n");
        return -1;
    }
    
    if (!rfc_path) {
        printf("[-] WARNING: No RFC specification file provided, skipping LLM analysis\n");
        return -1;
    }
    
    printf("\n[*] Starting RFC consistency analysis...\n");
    printf("    - RFC file: %s\n", rfc_path);
    printf("    - Protocol: %s\n", protocol_name ? protocol_name : "Unknown");
    printf("    - protocol parsed files: %s\n", output_dir);
    
    // Read RFC content
    char *rfc_content = read_entire_file(rfc_path);
    if (!rfc_content) {
        printf("[-] ERROR: Failed to read RFC specification file\n");
        return -1;
    }
    
    // Open interactions directory
    DIR *interactions_dir = opendir(rfc_interactions_dir);
    if (!interactions_dir) {
        printf("[-] ERROR: Cannot open interactions directory '%s': %s\n", 
               rfc_interactions_dir, strerror(errno));
        free(rfc_content);
        return -1;
    }
    
    // Process each interaction file
    struct dirent *entry;
    unsigned int processed_count = 0;
    unsigned int success_count = 0;
    
    while ((entry = readdir(interactions_dir)) != NULL) {
        // Check if this is an interaction file
        if (strstr(entry->d_name, "interaction-") && strstr(entry->d_name, ".jsonl")) {
            char *interaction_file_path = malloc(strlen(rfc_interactions_dir) + strlen(entry->d_name) + 16);
            if (!interaction_file_path) {
                printf("[-] ERROR: Memory allocation failed for interaction file path\n");
                continue;
            }
            snprintf(interaction_file_path, strlen(rfc_interactions_dir) + strlen(entry->d_name) + 16,
                    "%s/%s", rfc_interactions_dir, entry->d_name);
            
            // Extract interaction ID from filename
            char interaction_id[32];
            if (sscanf(entry->d_name, "interaction-%31s.jsonl", interaction_id) == 1) {
                // Remove extension if present
                char *dot = strchr(interaction_id, '.');
                if (dot) *dot = '\0';
                
                // Analyze this interaction
                if (analyze_single_interaction_file(interaction_file_path, interaction_id, rfc_content, protocol_name, output_dir) == 0) {
                    success_count++;
                }
                processed_count++;
                
                // Progress reporting
                if (processed_count % 5 == 0) {
                    printf("[*] Progress: %u interactions processed, %u successful\n", 
                           processed_count, success_count);
                }
            }
            
            free(interaction_file_path);
        }
    }
    
    closedir(interactions_dir);
    free(rfc_content);
    
    printf("[*] RFC consistency analysis completed!\n");
    printf("    - Total interactions processed: %u\n", processed_count);
    printf("    - Successful analyses: %u\n", success_count);
    printf("    - Results saved in: %s\n", rfc_llm_analysis_dir);
    
    return processed_count;
}


//===============================================following are the functions test===============================================
#ifdef TEST_PROTOCOL_ANALYSIS
int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <protocol_name> <spec_file_path> <output_dir>\n", argv[0]);
        fprintf(stderr, "Example: %s MODBUS modbus_spec.md ./output\n", argv[0]);
        return 1;
    }

    const char *protocol_name = argv[1];
    const char *spec_path = argv[2];
    const char *output_dir = argv[3];

    printf("Protocol Analysis Test\n");
    printf("=====================\n");
    printf("Protocol: %s\n", protocol_name);
    printf("Spec file: %s\n", spec_path);
    printf("Output directory: %s\n\n", output_dir);

    // Test protocol message grammar generation
    printf("Step 1: Generating protocol message grammar...\n");
    char *grammar_prompt = construct_prompt_for_message_grammar(protocol_name, spec_path);
    if (!grammar_prompt) {
        fprintf(stderr, "Error: Failed to construct grammar prompt\n");
        return 1;
    }

    printf("Grammar prompt constructed successfully\n");
    printf("Sending request to LLM API...\n");
    
    char *grammar_response = chat_with_llm(grammar_prompt, 3, 0.5);
    free(grammar_prompt);
    
    if (!grammar_response) {
        fprintf(stderr, "Error: Failed to get grammar response from LLM\n");
        return 1;
    }

    printf("LLM response received successfully\n");
    printf("Parsing and saving message grammar...\n");
    
    parse_and_save_message_grammar(grammar_response, protocol_name, output_dir);
    free(grammar_response);
    
    printf("Message grammar saved successfully\n\n");

    // Test protocol state machine generation
    printf("Step 2: Generating protocol state machine...\n");
    char *fsm_prompt = construct_prompt_for_state_machine(protocol_name, spec_path);
    if (!fsm_prompt) {
        fprintf(stderr, "Error: Failed to construct state machine prompt\n");
        return 1;
    }

    printf("State machine prompt constructed successfully\n");
    printf("Sending request to LLM API...\n");
    
    char *fsm_response = chat_with_llm(fsm_prompt, 3, 0.5);
    free(fsm_prompt);
    
    if (!fsm_response) {
        fprintf(stderr, "Error: Failed to get state machine response from LLM\n");
        return 1;
    }

    printf("LLM response received successfully\n");
    printf("Saving state machine...\n");
    
    save_state_machine(fsm_response, protocol_name, output_dir);
    free(fsm_response);
    
    printf("State machine saved successfully\n\n");
    
    // Verify the generated artifacts
    printf("Step 3: Verifying protocol analysis artifacts...\n");
    char *verification_result = NULL;
    bool verification_status = verify_protocol_analysis(protocol_name, output_dir, spec_path, &verification_result);
    
    if (!verification_status && verification_result) {
        // If issues were found, try to improve the artifacts
        printf("Step 4: Improving protocol analysis artifacts...\n");
        bool improvement_status = improve_protocol_analysis(protocol_name, output_dir, verification_result);
        
        if (improvement_status) {
            printf("Protocol analysis artifacts were improved successfully!\n");
        } else {
            printf("Failed to improve protocol analysis artifacts.\n");
        }
    }
    
    free(verification_result);
    
    printf("Test completed%s!\n", verification_status ? " successfully" : " with issues that were addressed");
    printf("Output files:\n");
    printf("  - %s/%s_message.json: Protocol message grammar in JSON format\n", output_dir, protocol_name);
    printf("  - %s/%s_message.txt: Additional protocol information\n", output_dir, protocol_name);
    printf("  - %s/%s_fsm.mmd: Protocol state machine in Mermaid format\n", output_dir, protocol_name);
    printf("  - %s/%s_verification.json: Verification results\n", output_dir, protocol_name);

    return 0;
}
#endif // TEST_PROTOCOL_ANALYSIS

#ifdef TEST_PROTOCOL_ANALYSIS_AND_UPDATE
int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <protocol_name> <spec_file_path> <output_dir>\n", argv[0]);
        fprintf(stderr, "Example: %s MODBUS modbus_spec.md ./output\n", argv[0]);
        return 1;
    }

    const char *protocol_name = argv[1];
    const char *spec_path = argv[2];
    const char *output_dir = argv[3];

    printf("Protocol Analysis and Update Test\n");
    printf("=================================\n");
    printf("Protocol: %s\n", protocol_name);
    printf("Spec file: %s\n", spec_path);
    printf("Output directory: %s\n\n", output_dir);

    // Phase 1: Initial Protocol Analysis (same as TEST_PROTOCOL_ANALYSIS)
    printf("=== PHASE 1: INITIAL PROTOCOL ANALYSIS ===\n\n");

    // Step 1: Generate protocol message grammar
    printf("Step 1: Generating protocol message grammar...\n");
    char *grammar_prompt = construct_prompt_for_message_grammar(protocol_name, spec_path);
    if (!grammar_prompt) {
        fprintf(stderr, "Error: Failed to construct grammar prompt\n");
        return 1;
    }

    printf("Grammar prompt constructed successfully\n");
    printf("Sending request to LLM API...\n");
    
    char *grammar_response = chat_with_llm(grammar_prompt, 3, 0.5);
    free(grammar_prompt);
    
    if (!grammar_response) {
        fprintf(stderr, "Error: Failed to get grammar response from LLM\n");
        return 1;
    }

    printf("LLM response received successfully\n");
    printf("Parsing and saving message grammar...\n");
    
    parse_and_save_message_grammar(grammar_response, protocol_name, output_dir);
    free(grammar_response);
    
    printf("Message grammar saved successfully\n\n");

    // Step 2: Generate protocol state machine
    printf("Step 2: Generating protocol state machine...\n");
    char *fsm_prompt = construct_prompt_for_state_machine(protocol_name, spec_path);
    if (!fsm_prompt) {
        fprintf(stderr, "Error: Failed to construct state machine prompt\n");
        return 1;
    }

    printf("State machine prompt constructed successfully\n");
    printf("Sending request to LLM API...\n");
    
    char *fsm_response = chat_with_llm(fsm_prompt, 3, 0.5);
    free(fsm_prompt);
    
    if (!fsm_response) {
        fprintf(stderr, "Error: Failed to get state machine response from LLM\n");
        return 1;
    }

    printf("LLM response received successfully\n");
    printf("Saving state machine...\n");
    
    save_state_machine(fsm_response, protocol_name, output_dir);
    free(fsm_response);
    
    printf("State machine saved successfully\n\n");

    // Phase 2: Verification and Update
    printf("=== PHASE 2: VERIFICATION AND UPDATE ===\n\n");

    // Step 3: Verify and update protocol grammar (JSON + TXT)
    printf("Step 3: Verifying and updating protocol grammar...\n");
    bool grammar_updated = verify_and_update_protocol_grammar(protocol_name, output_dir, spec_path);
    
    if (grammar_updated) {
        printf("Protocol grammar was updated successfully!\n");
        printf("   - Check new_%s_message.json for updated JSON grammar\n", protocol_name);
        printf("   - Check new_%s_message.txt for updated additional information\n", protocol_name);
    } else {
        printf("Protocol grammar verification passed - no updates needed\n");
    }
    printf("\n");

    // Step 4: Verify and update protocol state machine (MMD)
    printf("Step 4: Verifying and updating protocol state machine...\n");
    bool state_machine_updated = verify_and_update_protocol_state_machine(protocol_name, output_dir, spec_path);
    
    if (state_machine_updated) {
        printf("Protocol state machine was updated successfully!\n");
        printf("   - Check new_%s_fsm.mmd for updated state machine\n", protocol_name);
    } else {
        printf("Protocol state machine verification passed - no updates needed\n");
    }
    printf("\n");

    // Summary
    printf("=== SUMMARY ===\n");
    printf("Protocol analysis and verification completed!\n\n");
    
    printf("Generated files:\n");
    printf("%s/%s_message.json: Protocol message grammar (JSON)\n", output_dir, protocol_name);
    printf("%s/%s_message.txt: Additional protocol constraints (TXT)\n", output_dir, protocol_name);
    printf("%s/%s_fsm.mmd: Protocol state machine (Mermaid)\n", output_dir, protocol_name);
    
    if (grammar_updated || state_machine_updated) {
        printf("\nUpdated files:\n");
        if (grammar_updated) {
            printf("%s/new_%s_message.json: Updated protocol grammar (JSON)\n", output_dir, protocol_name);
            printf("%s/new_%s_message.txt: Updated additional information (TXT)\n", output_dir, protocol_name);
        }
        if (state_machine_updated) {
            printf("%s/new_%s_fsm.mmd: Updated state machine (Mermaid)\n", output_dir, protocol_name);
        }
        printf("\nThe 'new_' prefixed files contain the improved versions based on LLM verification.\n");
    }

    printf("\nTest completed successfully!\n");
    
    return 0;
}
#endif // TEST_PROTOCOL_ANALYSIS_AND_UPDATE

#ifdef TEST_ENRICH_INITIAL_SEEDS
#include <sys/stat.h>
#include <unistd.h>

// External variables that enrich_initial_seeds() depends on (normally defined in afl-fuzz.c)
char *protocol_name = NULL;
char *in_dir = NULL;
char *rfc_path = NULL;
char *parse_result_path = NULL;

// Simple implementation of enrich_initial_seeds for testing
void enrich_initial_seeds(void) {
    char *seed_question = NULL;
    const char *seedfile_path = in_dir;

    // 1. Generate the initial seed of message granularity
    if (rfc_path == NULL) {
        printf("RFC path is NULL, skipping LLM interaction\n");
        return;
    }

    printf("Step 1: Generating the initial seed of message granularity...\n");
    char *seeds_prompt = construct_prompt_for_seeds_message(protocol_name, &seed_question, seedfile_path, rfc_path);
    if (seeds_prompt == NULL) {
        printf("Failed to retrieve seeds prompt\n");
        FATAL("Failed to retrieve seeds prompt");
        return;
    }

    // Call the real LLM for message seeds generation
    printf("Calling LLM for message seeds generation...\n");
    char *seeds_answer = chat_with_llm(seeds_prompt, 3, 0.5);
    free(seeds_prompt);
    
    if (seeds_answer == NULL) {
        printf("Failed to get seeds from LLM\n");
        FATAL("Failed to get seeds from LLM");
        return;
    }
    printf("LLM Response for message seeds:\n");
    printf("================================\n");
    printf("%s\n", seeds_answer);
    printf("================================\n\n");

    // Extract sequences from the LLM output
    int num_sequences = 0;
    char **sequences = extract_sequences(seeds_answer, &num_sequences);
    free(seeds_answer);

    if (sequences != NULL && num_sequences > 0) {
        printf("Successfully extracted %d sequences from LLM output\n", num_sequences);
        
        // Display extracted sequences in hex format
        printf("Extracted sequences (hex format):\n");
        for (int i = 0; i < num_sequences; i++) {
            printf("  Sequence %d: ", i + 1);
            size_t len = (i < current_sequences_count) ? sequence_lengths[i] : 0;
            for (size_t j = 0; j < len; j++) {
                printf("%02X ", (unsigned char)sequences[i][j]);
            }
            printf("\n");
        }
        printf("\n");
        
        // Write the extracted sequences to seed files
        write_sequences_to_seeds(seedfile_path, sequences, num_sequences);

        // Free the allocated memory for sequences
        for (int i = 0; i < num_sequences; i++) {
            ck_free(sequences[i]);
        }
        ck_free(sequences);
    } else {
        printf("Warning: No valid sequences extracted from LLM output\n");
        printf("This might indicate that the LLM response doesn't contain properly formatted <sequence> tags\n\n");
    }

    // 2. Generate the initial seed of sequence granularity
    printf("\nStep 2: Generating the initial seed of sequence granularity...\n");
    char *sequences_prompt = construct_prompt_for_seeds_sequence(protocol_name, &seed_question, rfc_path);
    if (sequences_prompt == NULL) {
        printf("Failed to retrieve sequences prompt\n");
        FATAL("Failed to retrieve sequences prompt");
        return;
    }

    // Call the real LLM for sequence seeds generation
    printf("Calling LLM for sequence seeds generation...\n");
    char *sequences_answer = chat_with_llm(sequences_prompt, 3, 0.5);
    free(sequences_prompt);
    
    if (sequences_answer == NULL) {
        printf("Failed to get sequences from LLM\n");
        FATAL("Failed to get sequences from LLM");
        return;
    }
    printf("LLM Response for sequence seeds:\n");
    printf("=================================\n");
    printf("%s\n", sequences_answer);
    printf("=================================\n\n");

    // Extract messages from the LLM output
    int num_messages = 0;
    char **messages = extract_messages(sequences_answer, &num_messages);
    free(sequences_answer);

    if (messages != NULL && num_messages > 0) {
        printf("Successfully extracted %d sequences from LLM output\n", num_messages);
        
        // Display extracted message sequences in hex format
        printf("Extracted sequences (hex format):\n");
        for (int i = 0; i < num_messages; i++) {
            printf("  Sequence %d: ", i + 1);
            size_t len = (i < current_messages_count) ? message_lengths[i] : 0;
            for (size_t j = 0; j < len; j++) {
                printf("%02X ", (unsigned char)messages[i][j]);
            }
            printf("\n");
        }
        printf("\n");
        
        // Write the extracted messages to seed files
        write_messages_to_seeds(seedfile_path, messages, num_messages);

        // Free the allocated memory for messages
        for (int i = 0; i < num_messages; i++) {
            free(messages[i]);
        }
        free(messages);
    } else {
        printf("Warning: No valid sequences extracted from LLM output\n");
        printf("This might indicate that the LLM response doesn't contain properly formatted <sequence><message> tags\n\n");
    }

    // Clean up the global length arrays used for binary data tracking
    cleanup_length_arrays();
    
    printf("enrich_initial_seeds() completed successfully!\n");
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <protocol_name> <in_dir> <rfc_path> <parse_result_path>\n", argv[0]);
        fprintf(stderr, "Example: %s MODBUS ./seeds modbus_spec.md ./modbus-parse-result\n", argv[0]);
        return 1;
    }

    // Set global variables that enrich_initial_seeds() depends on
    protocol_name = strdup(argv[1]);
    in_dir = argv[2];
    rfc_path = argv[3];
    parse_result_path = argv[4];
    
    printf("Testing enrich_initial_seeds() function\n");
    printf("========================================\n");
    printf("Protocol: %s\n", protocol_name);
    printf("Seed directory: %s\n", in_dir);
    printf("RFC path: %s\n", rfc_path);
    printf("Parse result path: %s\n\n", parse_result_path);

    // Create seed directory if it doesn't exist
    struct stat st = {0};
    if (stat(in_dir, &st) == -1) {
        if (mkdir(in_dir, 0777) != 0) {
            fprintf(stderr, "Error: Failed to create seed directory '%s'\n", in_dir);
            return 1;
        }
        printf("Created seed directory: %s\n", in_dir);
    }

    // Check if RFC file exists
    if (access(rfc_path, F_OK) != 0) {
        fprintf(stderr, "Error: RFC file '%s' does not exist\n", rfc_path);
        return 1;
    }

    // Create parse result directory if it doesn't exist
    if (stat(parse_result_path, &st) == -1) {
        if (mkdir(parse_result_path, 0777) != 0) {
            fprintf(stderr, "Error: Failed to create parse result directory '%s'\n", parse_result_path);
            return 1;
        }
        printf("Created parse result directory: %s\n", parse_result_path);
    }

    printf("\nTest setup completed. Now calling enrich_initial_seeds()...\n");
    printf("=========================================================\n\n");

    // Check if OPENAI_API_KEY is set
    const char* api_key = getenv("LLM_API_KEY");
    if (!api_key) {
        fprintf(stderr, "Error: LLM_API_KEY environment variable is not set!\n");
        return 1;
    }

    // Call the function under test
    printf("Starting enrich_initial_seeds() with real LLM calls...\n");
    enrich_initial_seeds();

    printf("\nenrich_initial_seeds() test completed!\n");
    printf("Check the seed directory '%s' for generated seed files.\n", in_dir);

    // Cleanup
    free(protocol_name);

    return 0;
}
#endif // TEST_ENRICH_INITIAL_SEEDS

#ifdef TEST_EXTRACT_FUNCTIONS
// Test function specifically for extract_sequences and extract_messages
int test_extract_functions() {
    printf("Testing extract_sequences and extract_messages functions\n");
    printf("=======================================================\n\n");

    // Test extract_sequences with mock LLM output
    const char *mock_sequences_output = 
        "Here are some test sequences:\n"
        "<sequence>1234 ABCD EF01</sequence>\n"
        "<sequence>5678 9ABC DEF0</sequence>\n"
        "<sequence>1111 2222 3333</sequence>\n"
        "End of sequences.";

    printf("Test 1: Testing extract_sequences()\n");
    printf("Mock input: %s\n\n", mock_sequences_output);
    
    int num_sequences = 0;
    char **sequences = extract_sequences(mock_sequences_output, &num_sequences);
    
    if (sequences && num_sequences > 0) {
        printf("Successfully extracted %d sequences:\n", num_sequences);
        for (int i = 0; i < num_sequences; i++) {
            printf("Sequence %d: ", i + 1);
            // Print as hex since it's binary data
            size_t len = strlen(sequences[i]); // Note: this might be problematic for binary data with null bytes
            for (size_t j = 0; j < len; j++) {
                printf("%02X ", (unsigned char)sequences[i][j]);
            }
            printf("\n");
        }
        
        // Cleanup
        for (int i = 0; i < num_sequences; i++) {
            ck_free(sequences[i]);
        }
        ck_free(sequences);
    } else {
        printf("Failed to extract sequences or no sequences found\n");
    }

    printf("\n");

    // Test extract_messages with mock LLM output
    const char *mock_messages_output = 
        "Here are some test message sequences:\n"
        "<sequence>\n"
        "  <message>1234 ABCD</message>\n"
        "  <message>EF01 2345</message>\n"
        "</sequence>\n"
        "<sequence>\n"
        "  <message>5678 9ABC</message>\n"
        "  <message>DEF0 1111</message>\n"
        "  <message>2222 3333</message>\n"
        "</sequence>\n"
        "End of message sequences.";

    printf("Test 2: Testing extract_messages()\n");
    printf("Mock input: %s\n\n", mock_messages_output);
    
    int num_messages = 0;
    char **messages = extract_messages(mock_messages_output, &num_messages);
    
    if (messages && num_messages > 0) {
        printf("Successfully extracted %d message sequences:\n", num_messages);
        for (int i = 0; i < num_messages; i++) {
            printf("Message sequence %d: ", i + 1);
            // Print as hex since it's binary data
            size_t len = strlen(messages[i]); // Note: this might be problematic for binary data with null bytes
            for (size_t j = 0; j < len; j++) {
                printf("%02X ", (unsigned char)messages[i][j]);
            }
            printf("\n");
        }
        
        // Cleanup
        for (int i = 0; i < num_messages; i++) {
            free(messages[i]);
        }
        free(messages);
    } else {
        printf("Failed to extract messages or no messages found\n");
    }

    printf("\nExtract functions test completed!\n");
    return 0;
}

int main() {
    return test_extract_functions();
}
#endif // TEST_EXTRACT_FUNCTIONS

#ifdef TEST_INITIAL_SEEDS_OPTIMIZATION
#include <sys/stat.h>
#include <unistd.h>

// AFL-ICS specific functions and variables that initial_seeds_optimization() depends on
char *in_dir = NULL;
u32 mem_limit = 50; // 50MB default
u32 exec_tmout = 1000; // 1000ms default  
u8 qemu_mode = 0;
char *orig_cmdline = NULL;

// Test version of initial_seeds_optimization function - same logic as afl-fuzz.c but with real afl-cmin execution
void initial_seeds_optimization(){
    
    if (!in_dir) {
        printf("[-] WARNING: Input directory not set, skipping seed optimization\n");
        return;
    }
    
    // Check if we have any seeds to optimize
    struct dirent **nl;
    int seed_count = scandir(in_dir, &nl, NULL, alphasort);
    if (seed_count <= 1) {
        printf("[*] Too few seeds (%d) for optimization, skipping afl-cmin\n", seed_count > 0 ? seed_count - 2 : 0);
        if (seed_count > 0) {
            for (int i = 0; i < seed_count; i++) free(nl[i]);
            free(nl);
        }
        return;
    }
    
    printf("[*] Optimizing initial seeds using afl-cmin...\n");
    
    // Create temporary directory for optimized seeds
    char *temp_dir = NULL;
    // Remove trailing slash from in_dir to ensure temp_dir is a sibling directory
    char *in_dir_copy = strdup(in_dir);
    size_t len = strlen(in_dir_copy);
    while (len > 0 && in_dir_copy[len-1] == '/') {
        in_dir_copy[len-1] = '\0';
        len--;
    }
    
    if (asprintf(&temp_dir, "%s_optimized", in_dir_copy) == -1) {
        printf("[!] FATAL: Unable to allocate memory for temp directory path\n");
        free(in_dir_copy);
        exit(1);
    }
    free(in_dir_copy);
    
    // Remove existing temp directory if it exists
    char rm_cmd[512];
    snprintf(rm_cmd, sizeof(rm_cmd), "rm -rf '%s'", temp_dir);
    system(rm_cmd);
    
    // Create the temporary directory
    if (mkdir(temp_dir, 0700) && errno != EEXIST) {
        printf("[!] FATAL: Unable to create temp directory '%s'\n", temp_dir);
        exit(1);
    }
    
    // Construct afl-cmin command using original command line
    char *cmin_cmd = NULL;
    char *cmd_prefix = NULL;
    
    // Build the command prefix (same logic as afl-fuzz.c)
    if (asprintf(&cmd_prefix, "afl-cmin -i '%s' -o '%s' -m %u -t %u%s%s --", 
        in_dir,           // input directory
        temp_dir,         // output directory  
        mem_limit,        // memory limit
        exec_tmout,       // timeout
        qemu_mode ? " -Q" : "",  // QEMU mode if enabled
        getenv("AFL_FORKSRV_INIT_TMOUT") ? " -f" : ""  // fork server timeout
    ) == -1) {
        printf("[!] FATAL: Unable to allocate memory for afl-cmin command prefix\n");
        exit(1);
    }
    
    // Use the original command line (without afl-fuzz specific options)
    if (asprintf(&cmin_cmd, "%s %s", cmd_prefix, orig_cmdline) == -1) {
        free(cmd_prefix);
        printf("[!] FATAL: Unable to allocate memory for full afl-cmin command\n");
        exit(1);
    }
    
    free(cmd_prefix);
    
    printf("[*] Running: %s\n", cmin_cmd);
    
    // Execute afl-cmin (real execution, not simulation)
    int ret = system(cmin_cmd);
    free(cmin_cmd);
    
    if (ret != 0) {
        printf("[-] WARNING: afl-cmin failed (exit code: %d), keeping original seeds\n", ret);
        // Clean up temp directory
        system(rm_cmd);
        free(temp_dir);
        if (seed_count > 0) {
            for (int i = 0; i < seed_count; i++) free(nl[i]);
            free(nl);
        }
        return;
    }
    
    printf("[*] afl-cmin completed successfully\n");
    
    // Count optimized seeds
    struct dirent **opt_nl;
    int opt_count = scandir(temp_dir, &opt_nl, NULL, alphasort);
    int original_seeds = seed_count - 2;  // Exclude . and ..
    int optimized_seeds = opt_count > 0 ? opt_count - 2 : 0;
    
    printf("[*] Original seeds: %d, Optimized seeds: %d\n", original_seeds, optimized_seeds);
    
    if (optimized_seeds <= 0) {
        printf("[-] WARNING: afl-cmin produced no seeds, keeping original seeds\n");
        // Clean up
        if (opt_count > 0) {
            for (int i = 0; i < opt_count; i++) free(opt_nl[i]);
            free(opt_nl);
        }
        system(rm_cmd);
        free(temp_dir);
        for (int i = 0; i < seed_count; i++) free(nl[i]);
        free(nl);
        return;
    }
    
    // Backup original seeds
    char *backup_dir = NULL;
    // Extract parent directory and create backup as sibling directory
    char *in_dir_backup = strdup(in_dir);
    // Remove trailing slash if present
    size_t backup_len = strlen(in_dir_backup);
    while (backup_len > 0 && in_dir_backup[backup_len-1] == '/') {
        in_dir_backup[backup_len-1] = '\0';
        backup_len--;
    }
    
    if (asprintf(&backup_dir, "%s_backup", in_dir_backup) == -1) {
        printf("[!] FATAL: Unable to allocate memory for backup directory path\n");
        free(in_dir_backup);
        exit(1);
    }
    free(in_dir_backup);
    
    // Remove existing backup directory and create backup
    char backup_cmd[512];
    snprintf(backup_cmd, sizeof(backup_cmd), "rm -rf '%s'", backup_dir);
    system(backup_cmd);
    
    snprintf(backup_cmd, sizeof(backup_cmd), "cp -r '%s' '%s'", in_dir, backup_dir);
    if (system(backup_cmd) != 0) {
        printf("[-] WARNING: Failed to backup original seeds\n");
    } else {
        printf("[*] Successfully backed up original seeds to %s\n", backup_dir);
    }
    
    // Replace original seeds with optimized ones
    // First, try to remove all files from the input directory
    char clear_cmd[512];
    snprintf(clear_cmd, sizeof(clear_cmd), "rm -f '%s'/*", in_dir);
    printf("[*] Clearing original seeds: %s\n", clear_cmd);
    system(clear_cmd);
    
    // Then copy optimized seeds to input directory
    char copy_cmd[512];
    snprintf(copy_cmd, sizeof(copy_cmd), "cp '%s'/* '%s'/ 2>/dev/null", temp_dir, in_dir);
    printf("[*] Copying optimized seeds: %s\n", copy_cmd);
    
    if (system(copy_cmd) != 0) {
        printf("[-] WARNING: Failed to copy optimized seeds\n");
        // Try to restore backup
        snprintf(copy_cmd, sizeof(copy_cmd), "cp '%s'/* '%s'/ 2>/dev/null", backup_dir, in_dir);
        if (system(copy_cmd) != 0) {
            printf("[-] CRITICAL: Failed to restore backup! Manual recovery needed.\n");
            printf("    Original seeds backup: %s\n", backup_dir);
            printf("    Optimized seeds: %s\n", temp_dir);
            printf("    You may need to manually copy files from backup to restore.\n");
        } else {
            printf("[*] Restored original seeds from backup\n");
        }
    } else {
        printf("[*] Seed optimization completed: %d → %d seeds (%.1f%% reduction)\n", 
             original_seeds, optimized_seeds, 
             100.0 * (original_seeds - optimized_seeds) / original_seeds);
    }
    
    // Clean up temporary directory
    char temp_cleanup_cmd[512];
    snprintf(temp_cleanup_cmd, sizeof(temp_cleanup_cmd), "rm -rf '%s'", temp_dir);
    system(temp_cleanup_cmd);
    
    // Only remove backup after successful replacement
    char backup_cleanup_cmd[512];
    snprintf(backup_cleanup_cmd, sizeof(backup_cleanup_cmd), "rm -rf '%s'", backup_dir);
    system(backup_cleanup_cmd);
    
    // Free allocated memory
    free(temp_dir);
    free(backup_dir);
    for (int i = 0; i < seed_count; i++) free(nl[i]);
    free(nl);
    if (opt_count > 0) {
        for (int i = 0; i < opt_count; i++) free(opt_nl[i]);
        free(opt_nl);
    }
}

// Helper function to create test seeds
void create_test_seeds(const char* seed_dir, int num_seeds) {
    printf("[*] Creating %d test seeds in %s...\n", num_seeds, seed_dir);
    
    // Create seed directory if it doesn't exist
    struct stat st = {0};
    if (stat(seed_dir, &st) == -1) {
        if (mkdir(seed_dir, 0777) != 0) {
            printf("Error: Failed to create seed directory '%s'\n", seed_dir);
            return;
        }
    }
    
    for (int i = 0; i < num_seeds; i++) {
        char *seed_filename = NULL;
        asprintf(&seed_filename, "%s/seed_%03d.bin", seed_dir, i + 1);
        
        FILE *fp = fopen(seed_filename, "wb");
        if (fp) {
            // Write some test data (simple protocol message pattern)
            unsigned char test_data[] = {
                0x68, 0x04,             // Start + Length  
                (unsigned char)(i & 0xFF), (unsigned char)((i >> 8) & 0xFF),  // Variable data based on index
                0x00, 0x00,             // Padding
                (unsigned char)(i * 2), (unsigned char)(i * 3),              // More variable data
                0x16                    // End marker
            };
            fwrite(test_data, sizeof(test_data), 1, fp);
            fclose(fp);
            printf("  Created: %s (%zu bytes)\n", seed_filename, sizeof(test_data));
        }
        
        free(seed_filename);
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <seed_dir> <target_command> [mem_limit_mb] [timeout_ms] [qemu_mode]\n", argv[0]);
        fprintf(stderr, "Example: %s ./test_seeds '/bin/cat @@' 50 1000 0\n", argv[0]);
        fprintf(stderr, "         %s ./test_seeds '/path/to/server 1502' 50 1000 0\n", argv[0]);
        fprintf(stderr, "         %s ./test_seeds 'echo Hello' 50 1000 0\n", argv[0]);
        fprintf(stderr, "\nNote: This test executes real afl-cmin, so:\n");
        fprintf(stderr, "      - Ensure afl-cmin is in your PATH\n");
        fprintf(stderr, "      - Provide a valid target command with all arguments\n");
        fprintf(stderr, "      - Use '@@' in target command for AFL input file substitution\n");
        fprintf(stderr, "      - Quote the target command if it contains spaces or arguments\n");
        return 1;
    }

    // Set up test parameters
    in_dir = argv[1];
    orig_cmdline = argv[2];  // Target binary is now required
    
    if (argc >= 4) {
        mem_limit = (u32)strtoul(argv[3], NULL, 10);
    }
    
    if (argc >= 5) {
        exec_tmout = (u32)strtoul(argv[4], NULL, 10);
    }
    
    if (argc >= 6) {
        qemu_mode = (u8)strtoul(argv[5], NULL, 10);
    }
    
    printf("Initial Seeds Optimization Test (Real afl-cmin)\n");
    printf("===============================================\n");
    printf("Seed directory: %s\n", in_dir);
    printf("Target command: %s\n", orig_cmdline);
    printf("Memory limit: %u MB\n", mem_limit);
    printf("Execution timeout: %u ms\n", exec_tmout);
    printf("QEMU mode: %s\n\n", qemu_mode ? "enabled" : "disabled");
    
    // Warn if memory limit seems too high (likely a misunderstood argument)
    if (mem_limit > 500) {
        printf("WARNING: Memory limit %u MB seems unusually high.\n", mem_limit);
        printf("         If you meant to pass a port number or other argument,\n");
        printf("         include it in the target command instead: '%s <port>'\n", orig_cmdline);
        printf("         Example: '/path/to/server 1502' instead of separate arguments\n\n");
    }
    
    // Check if afl-cmin is available
    printf("Checking for afl-cmin availability...\n");
    if (system("which afl-cmin > /dev/null 2>&1") != 0) {
        fprintf(stderr, "Error: afl-cmin not found in PATH\n");
        fprintf(stderr, "Please install AFL or ensure afl-cmin is available\n");
        return 1;
    }
    printf("✓ afl-cmin found\n");

    // Check if seed directory exists
    struct stat st = {0};
    if (stat(in_dir, &st) == -1) {
        printf("Seed directory doesn't exist. Creating test seeds...\n");
        create_test_seeds(in_dir, 10);  // Create 10 test seeds
    } else {
        printf("Using existing seed directory.\n");
    }
    
    // Count initial seeds
    struct dirent **nl_before;
    int seed_count_before = scandir(in_dir, &nl_before, NULL, alphasort);
    int seeds_before = seed_count_before > 0 ? seed_count_before - 2 : 0;  // Exclude . and ..
    
    printf("Seeds before optimization: %d\n", seeds_before);
    
    if (seed_count_before > 0) {
        printf("Seed files:\n");
        for (int i = 0; i < seed_count_before; i++) {
            if (strcmp(nl_before[i]->d_name, ".") != 0 && strcmp(nl_before[i]->d_name, "..") != 0) {
                printf("  - %s\n", nl_before[i]->d_name);
            }
            free(nl_before[i]);
        }
        free(nl_before);
    }
    
    printf("\nRunning initial_seeds_optimization()...\n");
    printf("=====================================\n");
    
    // Call the function under test
    initial_seeds_optimization();
    
    // Count seeds after optimization
    struct dirent **nl_after;
    int seed_count_after = scandir(in_dir, &nl_after, NULL, alphasort);
    int seeds_after = seed_count_after > 0 ? seed_count_after - 2 : 0;  // Exclude . and ..
    
    printf("\nResults:\n");
    printf("========\n");
    printf("Seeds after optimization: %d\n", seeds_after);
    
    if (seed_count_after > 0) {
        printf("Remaining seed files:\n");
        for (int i = 0; i < seed_count_after; i++) {
            if (strcmp(nl_after[i]->d_name, ".") != 0 && strcmp(nl_after[i]->d_name, "..") != 0) {
                struct stat file_stat;
                char *full_path = NULL;
                asprintf(&full_path, "%s/%s", in_dir, nl_after[i]->d_name);
                
                if (stat(full_path, &file_stat) == 0) {
                    printf("  - %s (%ld bytes)\n", nl_after[i]->d_name, file_stat.st_size);
                } else {
                    printf("  - %s (size unknown)\n", nl_after[i]->d_name);
                }
                
                free(full_path);
            }
            free(nl_after[i]);
        }
        free(nl_after);
    }
    
    printf("\nOptimization Summary:\n");
    printf("====================\n");
    if (seeds_before > 0) {
        printf("Original seeds: %d\n", seeds_before);
        printf("Optimized seeds: %d\n", seeds_after);
        printf("Reduction: %d seeds (%.1f%%)\n", 
               seeds_before - seeds_after,
               seeds_before > 0 ? 100.0 * (seeds_before - seeds_after) / seeds_before : 0.0);
    } else {
        printf("No seeds to optimize.\n");
    }
    
    printf("\nTest completed successfully!\n");
    printf("Check directory '%s' for optimized seeds.\n", in_dir);
    printf("Note: Temporary files and backup directories have been automatically cleaned up.\n");
    
    return 0;
}
#endif // TEST_INITIAL_SEEDS_OPTIMIZATION

//RFC Consistency Analysis Test
#ifdef TEST_RFC_CONSISTENCY
int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <output_dir> [rfc_path]\n", argv[0]);
        printf("Example: %s ./test_output ./sample_rfc.md\n", argv[0]);
        return 1;
    }
    
    const char *output_dir = argv[1];
    const char *rfc_path = (argc > 2) ? argv[2] : NULL;
    
    printf("RFC Consistency Analysis Test\n");
    printf("=============================\n");
    printf("Output directory: %s\n", output_dir);
    printf("RFC file: %s\n", rfc_path ? rfc_path : "None provided");
    
    // Test initialization
    printf("\n[Test 1] Initializing RFC consistency analysis...\n");
    if (init_rfc_consistency_analysis(output_dir) != 0) {
        printf("FAILED: Initialization failed\n");
        return 1;
    }
    printf("SUCCESS: Initialization completed\n");
    
    // Test mode setting
    printf("\n[Test 2] Setting collection mode...\n");
    set_rfc_consistency_mode(COLLECT_WITH_SAMPLING, 50);
    printf("SUCCESS: Mode set to COLLECT_WITH_SAMPLING with 50‰ sampling\n");
    
    // Test data collection decision
    printf("\n[Test 3] Testing collection decision logic...\n");
    printf("should_collect_consistency_data(2): %s\n", 
           should_collect_consistency_data(2) ? "YES" : "NO");
    printf("should_collect_consistency_data(1): %s\n", 
           should_collect_consistency_data(1) ? "YES" : "NO");
    printf("should_collect_consistency_data(0): %s (sampling-dependent)\n", 
           should_collect_consistency_data(0) ? "YES" : "NO");
    
    // Test data collection (with dummy data)
    printf("\n[Test 4] Testing data collection...\n");
    printf("SKIPPED: Data collection test requires valid kl_messages structure\n");
    printf("NOTE: response_time_us field removed - not reliably available in AFL-ICS\n");
    
    // Test RFC analysis (if RFC file provided)
    if (rfc_path) {
        printf("\n[Test 5] Testing RFC consistency analysis...\n");
        int result = perform_rfc_consistency_analysis(rfc_path, "TEST_PROTOCOL");
        if (result >= 0) {
            printf("SUCCESS: RFC analysis completed (%d interactions processed)\n", result);
        } else {
            printf("FAILED: RFC analysis failed\n");
        }
    } else {
        printf("\n[Test 5] Skipping RFC analysis (no RFC file provided)\n");
    }
    
    // Test cleanup
    printf("\n[Test 6] Testing cleanup...\n");
    cleanup_rfc_consistency_analysis();
    printf("SUCCESS: Cleanup completed\n");
    
    printf("\n=== All tests completed ===\n");
    return 0;
}
#endif // TEST_RFC_CONSISTENCY


