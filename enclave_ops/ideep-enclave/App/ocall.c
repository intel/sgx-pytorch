/*
 * Copyright (C) 2011-2021 Intel Corporation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the
 *     distribution.
 *   * Neither the name of Intel Corporation nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>

#include <stdbool.h>

typedef enum _ra_msg_type_t
{
     TYPE_RA_MSG0 = 0,
     TYPE_RA_MSG1,
     TYPE_RA_MSG2,
     TYPE_RA_MSG3,
     TYPE_RA_ATT_RESULT,
     TYPE_RA_RETRIEVE_DK,
}ra_msg_type_t;

#ifndef SAFE_FREE
#define SAFE_FREE(ptr) {if (NULL != (ptr)) {free(ptr); (ptr) = NULL;}}
#endif

const char prov_ip_addr[] = "127.0.0.1";
const uint32_t prov_port = 8887;

#pragma pack(1)

typedef struct _ra_samp_request_header_t{
    uint8_t  type;     /* set to one of ra_msg_type_t*/
    uint32_t size;     /*size of request body*/
    uint32_t model_id;
    uint8_t  align[3];
    uint8_t body[];
} ra_samp_request_header_t;

typedef struct _ra_samp_response_header_t{
    uint8_t  type;      /* set to one of ra_msg_type_t*/
    uint8_t  status[2];
    uint32_t size;      /*size of the response body*/
    uint8_t  align[1];
    uint8_t  body[];
} ra_samp_response_header_t;

typedef struct sample_key_blob_t {
    uint32_t        blob_size;
    uint8_t         blob[];
} sample_key_blob_t;

#pragma pack()


static int32_t g_sock = -1;

static void Connect()
{
    int32_t retry_count = 60;
    struct sockaddr_in serAddr;
    int32_t sockFd = -1;

    sockFd = socket(AF_INET, SOCK_STREAM, 0);
    if(sockFd < 0) {
        printf("Create socket failed\n");
        exit(1);
    }
    bzero(&serAddr, sizeof(serAddr));
    serAddr.sin_family = AF_INET;
    serAddr.sin_port = htons(prov_port);
    serAddr.sin_addr.s_addr = inet_addr(prov_ip_addr);

    do {
        if(connect(sockFd, (struct sockaddr*)&serAddr, sizeof(serAddr)) >= 0) {
            printf("Connect socket server suucess!\n");
            break;
        }
        else if (retry_count > 0) {
            printf("Connect socket server failed, sleep 0.5s and try again...\n");
            usleep(500000); // 0.5 s
        }
        else {
            printf("Fail to connect socket server.\n");
            close(sockFd);
           return;
        }
    } while (retry_count-- > 0);

    g_sock = sockFd;
}


bool IsConnected()
{
    if (g_sock > 0)
        return true;
    else
        return false;
}

static void Disconnect()
{
    close(g_sock);
    g_sock = -1;
}

static bool SendAll(int32_t sock, const void *data, int32_t data_size)
{
    const char *data_ptr = (const char*) data;
    int32_t bytes_sent;

    while (data_size > 0)
    {
        bytes_sent = send(sock, data_ptr, data_size, 0);
        if (bytes_sent < 1)
            return false;

        data_ptr += bytes_sent;
        data_size -= bytes_sent;
    }

    return true;
}


static bool RecvAll(int32_t sock, void *data, int32_t data_size)
{
    char *data_ptr = (char*) data;
    int32_t bytes_recv;

    while (data_size > 0)
    {
        bytes_recv = recv(sock, data_ptr, data_size, 0);
        if (bytes_recv == 0) {
            printf("the server side may closed...\n");
            return true;
        }
        if (bytes_recv < 0) {
            printf("failed to read data\n");
            return false;
        }

        data_ptr += bytes_recv;
        data_size -= bytes_recv;
    }

    return true;
}

static int SendAndRecvMsg(const ra_samp_request_header_t *p_req,
    ra_samp_response_header_t **p_resp)
{
    ra_samp_response_header_t* out_msg;
    int req_size, resp_size = 0;
    int err = 0;

    if((NULL == p_req) ||
        (NULL == p_resp))
    {
        return -1;
    }

    /* Send a message to server */
    req_size = sizeof(ra_samp_request_header_t)+p_req->size;

    if (!SendAll(g_sock, &req_size, sizeof(req_size))) {
        printf("send req_size failed\n");
        err = -1;
        goto out;
    }
    if (!SendAll(g_sock, p_req, req_size)) {
        printf("send req buffer failed\n");
        err = -1;
        goto out;
    }

    /* Receive a message from server */
    if (!RecvAll(g_sock, &resp_size, sizeof(resp_size))) {
        printf("failed to get the resp size\n");
        err = -1;
        goto out;
    }

    if (resp_size <= 0) {
        printf("no msg need to read\n");
        err = -1;
        goto out;
    }
    out_msg = (ra_samp_response_header_t *)malloc(resp_size);
    if (!out_msg) {
        printf("allocate out_msg failed\n");
        err = -1;
        goto out;
    }
    if (!RecvAll(g_sock, out_msg, resp_size)) {
        printf("failed to get the data\n");
        err = -1;
        goto out;
    }

    *p_resp = out_msg;
out:
    return err;
}

int RetrieveDomainKey(uint32_t model_id, uint8_t *key_blob, uint32_t key_blob_size)
{

printf("hyhyhyh RetrieveDomainKey called.\n");

//    SgxCrypto::EnclaveHelpers enclaveHelpers;
    ra_samp_request_header_t *p_req = NULL;
    ra_samp_response_header_t *p_resp = NULL;
    sample_key_blob_t *p_dk = NULL;

    sgx_status_t sgxStatus = SGX_ERROR_UNEXPECTED;
    sgx_status_t sgx_ret = SGX_ERROR_UNEXPECTED;

    int ret = 0;

    if (!IsConnected())
        Connect();

    p_req = (ra_samp_request_header_t *)malloc(sizeof(ra_samp_request_header_t));
    if (!p_req) {
        printf("allocate memory failed\n");
        ret = -1;
        goto out;
    }

    /* retrieve the domainkey blob through the socket */

    //no extra payload need to sent
    p_req->size = 0;
    p_req->model_id = model_id;
    p_req->type = TYPE_RA_RETRIEVE_DK;
    SendAndRecvMsg(p_req, &p_resp);

    if (!p_resp || (p_resp->status[0] != 0) || (p_resp->status[1] != 0)) {
        printf("failed to get the resp message.\n");
        ret = -1;
        goto out;
    }

    if (TYPE_RA_RETRIEVE_DK != p_resp->type) {
        printf("the resp msg type is not matched.\n");
        ret = -1;
        goto out;
    }

    p_dk = (sample_key_blob_t*)p_resp->body;

    if (p_dk->blob_size != key_blob_size) {
        printf("hyhyhy ocall key_blob_size(%d) does not equal to p_dk->blob_size(%d).\n", key_blob_size ,p_dk->blob_size);
        goto out;
    }
printf("hyhyhy p_dk->blob_size is %d.\n ", p_dk->blob_size);

    memcpy(key_blob, p_dk->blob, p_dk->blob_size);
printf("hyhyhy2222222  p_dk->blob_size is %d.\n ", p_dk->blob_size);

out:
    if (IsConnected())
        Disconnect();

    SAFE_FREE(p_req);
    SAFE_FREE(p_resp);

    return ret;
}

void ocall_print_string(const char *str)
{
    printf("%s", str);
}

int ocall_get_encryption_key_blob(uint32_t model_id, uint8_t *key_blob, uint32_t key_blob_size)
{
 
printf("hyhyhy ocall_get_encryption_key_blob called, model_id is 0x%d, key_blob is %p, key_blob_size is %d.\n", model_id, key_blob, key_blob_size);
    int ret = RetrieveDomainKey(model_id, key_blob, key_blob_size);
printf("RetrieveDomainKey ret is %d.\n", ret);

    return ret;
}




