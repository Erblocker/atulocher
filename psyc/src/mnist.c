/*
 Copyright (c) 2016 Fabio Nicotra.
 All rights reserved.
 
 Redistribution and use in source and binary forms are permitted
 provided that the above copyright notice and this paragraph are
 duplicated in all such forms and that any documentation,
 advertising materials, and other materials related to such
 distribution and use acknowledge that the software was developed
 by the copyright holder. The name of the
 copyright holder may not be used to endorse or promote products derived
 from this software without specific prior written permission.
 THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
 IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <zlib.h>
#include "mnist.h"

#define CHUNK 16384
#define IMAGES_MAGIC_NUM 2051
#define LABELS_MAGIC_NUM 2049
#define le2be(x) ((x >> 24 & 0x000000FF) | \
    (x >> 8 & 0x0000FF00) | \
    (x << 8 & 0x00FF0000) | \
    (x << 24))

int decompressGZip(FILE *source, FILE * dest) {
    int ret;
    unsigned have;
    z_stream strm;
    unsigned char in[CHUNK];
    unsigned char out[CHUNK];
    
    /* allocate inflate state */
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    strm.avail_in = 0;
    strm.next_in = Z_NULL;
    ret = inflateInit2(&strm, 16+MAX_WBITS);
    if (ret != Z_OK)
        return ret;
    /* decompress until deflate stream ends or end of file */
    do {
        strm.avail_in = fread(in, 1, CHUNK, source);
        if (ferror(source)) {
            (void)inflateEnd(&strm);
            return Z_ERRNO;
        }
        if (strm.avail_in == 0)
            break;
        strm.next_in = in;
        /* run inflate() on input until output buffer not full */
        do {
            strm.avail_out = CHUNK;
            strm.next_out = out;
            ret = inflate(&strm, Z_NO_FLUSH);
            assert(ret != Z_STREAM_ERROR);  /* state not clobbered */
            switch (ret) {
                case Z_NEED_DICT:
                    ret = Z_DATA_ERROR;     /* and fall through */
                case Z_DATA_ERROR:
                case Z_MEM_ERROR:
                    (void)inflateEnd(&strm);
                    return ret;
            }
            have = CHUNK - strm.avail_out;
            if (fwrite(out, 1, have, dest) != have || ferror(dest)) {
                (void)inflateEnd(&strm);
                return Z_ERRNO;
            }
        } while (strm.avail_out == 0);
        /* done when inflate() says it's done */
    } while (ret != Z_STREAM_END);
    /* clean up and return */
    (void)inflateEnd(&strm);
    return ret == Z_STREAM_END ? Z_OK : Z_DATA_ERROR;
}


/* report a zlib or i/o error */
void zerr(int ret)
{
    fputs("zpipe: ", stderr);
    switch (ret) {
        case Z_ERRNO:
            if (ferror(stdin))
                fputs("error reading stdin\n", stderr);
            if (ferror(stdout))
                fputs("error writing stdout\n", stderr);
            break;
        case Z_STREAM_ERROR:
            fputs("invalid compression level\n", stderr);
            break;
        case Z_DATA_ERROR:
            fputs("invalid or incomplete deflate data\n", stderr);
            break;
        case Z_MEM_ERROR:
            fputs("out of memory\n", stderr);
            break;
        case Z_VERSION_ERROR:
            fputs("zlib version mismatch!\n", stderr);
    }
}

void getTempFileName(const char * prefix, char * buffer) {
    FILE * urand = fopen("/dev/urandom", "r");
    char buff[4];
    fgets(buff, 4, urand);
    sprintf(buffer, "/tmp/%s-%02x%02x%02x%02x",
            prefix,
            (unsigned char) buff[0],
            (unsigned char) buff[1],
            (unsigned char) buff[2],
            (unsigned char) buff[3]);
    fclose(urand);
}

int loadMNISTData(int type,
                  const char * images_file,
                  const char * labels_file,
                  double ** data) {
    char tmpImagesFileName[255];
    char tmpLabelsFileName[255];
    char * prefixImg;
    char * prefixLbl;
    int err;
    if (type == TRAINING_DATA) {
        printf("Loading MNIST Data for training...\n");
        prefixImg = "train-images";
        prefixLbl = "train-labels";
    } else {
        printf("Loading MNIST Data for testing...\n");
        prefixImg = "test-images";
        prefixLbl = "test-labels";
    }
    getTempFileName(prefixImg, tmpImagesFileName);
    getTempFileName(prefixLbl, tmpLabelsFileName);
    FILE * images = fopen(images_file, "r");
    FILE * labels = fopen(labels_file, "r");
    if (images == NULL) {
        fprintf(stderr, "Cannot open %s\n", images_file);
        data = NULL;
        return 0;
    }
    if (labels == NULL) {
        fprintf(stderr, "Cannot open %s\n", labels_file);
        data = NULL;
        return 0;
    }
    FILE * tmpimages = fopen(tmpImagesFileName, "w");
    FILE * tmplabels = fopen(tmpLabelsFileName, "w");
    printf("Loading images...\n");
    err = decompressGZip(images, tmpimages);
    if (err) {zerr(err); data = NULL; return 0;}
    printf("Loading labels...\n");
    err = decompressGZip(labels, tmplabels);
    if (err) {zerr(err); data = NULL; return 0;}
    fclose(tmpimages);
    fclose(tmplabels);
    tmpimages = fopen(tmpImagesFileName, "r");
    tmplabels = fopen(tmpLabelsFileName, "r");
    fseek(tmpimages, 0, SEEK_SET);
    fseek(tmplabels, 0, SEEK_SET);
    uint32_t magic_num = 0, image_count = 0, label_count = 0;
    int i = 0, j = 0;
    fread(&magic_num, 1, 4, tmpimages);
    magic_num = le2be(magic_num);
    if (magic_num != IMAGES_MAGIC_NUM) {
        fprintf(stderr, "Invalid magic number for image file: %d\n", magic_num);
        data = NULL;
        return 0;
    }
    fread(&image_count, 1, 4, tmpimages);
    image_count = le2be(image_count);
    if (image_count == 0) {
        fputs("Image count is 0!\n", stderr);
        data = NULL;
        return 0;
    }
    printf("Found %d images.\n", image_count);
    fread(&magic_num, 1, 4, tmplabels);
    magic_num = le2be(magic_num);
    if (magic_num != LABELS_MAGIC_NUM) {
        fprintf(stderr, "Invalid magic number for labels file: %d\n",magic_num);
        data = NULL;
        return 0;
    }
    fread(&label_count, 1, 4, tmplabels);
    label_count = le2be(label_count);
    if (label_count == 0) {
        fputs("Label count is 0!\n", stderr);
        data = NULL;
        return 0;
    }
    printf("Found %d labels.\n", label_count);
    if (label_count != image_count) {
        fputs("Image count and label count do not match!\n", stderr);
        data = NULL;
        return 0;
    }
    uint32_t rows = 0, cols = 0;
    fread(&rows, 1, 4, tmpimages);
    fread(&cols, 1, 4, tmpimages);
    rows = le2be(rows);
    cols = le2be(cols);
    printf("Image size: %dx%d\n", rows, cols);
    int img_area = rows * cols;
    if (img_area == 0) {
        fputs("Invalid image size!\n", stderr);
        data = NULL;
        return 0;
    }
    int data_len = (img_area * image_count) + (label_count * 10);
    *data = malloc(data_len * sizeof(double));
    double * data_p = *data;
    for (i = 0; i < (int) image_count; i++) {
        printf("\rLoading image %d/%d", i + 1, image_count);
        for (j = 0; j < img_area; j++) {
            int pixel = fgetc(tmpimages);
            double d = (double) pixel / (double) 255;
            *data_p = d;
            data_p++;
        }
        int label = fgetc(tmplabels);
        //printf("Label: %d", label);
        for (j = 0; j < 10; j++) {
            *data_p = (j == label);
            data_p++;
        }
    }
    printf("\n");
    fclose(images);
    fclose(labels);
    fclose(tmpimages);
    fclose(tmplabels);
    remove(tmpImagesFileName);
    remove(tmpLabelsFileName);
    //printf("Datalen: %d\n", data_len);
    //printf("Allocated data size: %d\n", data_p - *data);
    return data_len;
}

/*int main(int argc, char** argv) {
    //printf("Argc: %d\n", argc);
    double * training_data = NULL;
    loadMNISTData(TRAINING_DATA,
                  "train-images-idx3-ubyte.gz",
                  "train-labels-idx1-ubyte.gz",
                  &training_data);
    return 0;
}*/

