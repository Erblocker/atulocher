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
#include <math.h>
#include <wand/MagickWand.h>
#include "image_data.h"
#include "utils.h"

static ExceptionType serverity;
static int __magick_instantiated = 0; //Ensure compatibility with older versions

#define IS_OK(x) (x == MagickTrue) 
#define LOG_MAGICK_ERR(wand) fprintf(stderr, "Magick ERROR: %s\n", \
    MagickGetException(wand, &serverity))

/*static void dumpPixels(int w, int h, int bytes, double* pixels){
    int i, numsamples = 0, area = w * h * bytes;
    double last_pixel;
    for (i = 0; i < area; i++) {
        double pixel = pixels[i];
        if (i == 0 || last_pixel != pixel) {
            int x = i % w;
            int y = i / w;
            printf("PX[%d] (%d, %d): %lf\n", i, x, y, pixel);
            if (++numsamples > 4) return;
        }
        last_pixel = pixel;
    }
}*/

static float getScaleFactorToFit(int w, int h, int fitW, int fitH) {
    float sw = (float) fitW / (float) w;
    float sh = (float) fitH / (float) h;
    float scale = (sw > sh ? sw : sh);
    return (scale > 1 ? 1 : scale);
}

static double * getImagePixels(char * filename, int fit_w, int fit_h,
                               int grayscale, int invert, char* bgcolor,
                               char* dump_file) {
    //if (IsMagickWandInstantiated() != MagickTrue)
    if (!__magick_instantiated) {
        MagickWandGenesis();
        __magick_instantiated = 1;
    }
    MagickWand * wand;
    wand = NewMagickWand();
    
    MagickBooleanType ok = MagickReadImage(wand, filename);
    if (!IS_OK(ok)) {
        LOG_MAGICK_ERR(wand);
        DestroyMagickWand(wand);
        MagickWandTerminus();
        return NULL;
    }
    
    size_t w = MagickGetImageWidth(wand);
    size_t h = MagickGetImageHeight(wand);
    ColorspaceType colorspace = MagickGetImageColorspace(wand);
    /*size_t d = MagickGetImageDepth(wand);
    int has_alpha = MagickGetImageAlphaChannel(wand) == MagickTrue;
    ImageType imgtype = MagickGetImageType(wand);
    ColorspaceType colorspace = MagickGetImageColorspace(wand);*/
    
    int pixel_size;
    char * channel_map;
    
    if (grayscale) {
        pixel_size = 1;
        channel_map = "I";
        if (colorspace != GRAYColorspace) {
            ok = MagickSetImageType(wand, GrayscaleType);
            if (!IS_OK(ok)) {
                LOG_MAGICK_ERR(wand);
                DestroyMagickWand(wand);
                MagickWandTerminus();
                return NULL;
            } else {
                ok = MagickSetImageColorspace(wand, GRAYColorspace);
                if (!IS_OK(ok)) {
                    LOG_MAGICK_ERR(wand);
                    DestroyMagickWand(wand);
                    MagickWandTerminus();
                    return NULL;
                }
            }
        }
    } else {
        pixel_size = 4;
        channel_map = "RGBA";
    }
    
    float scale = getScaleFactorToFit(w, h, fit_w, fit_h);
    
    if (scale < 1) {
        w *= scale;
        h *= scale;
        ok = MagickScaleImage(wand, w, h);
        if (!IS_OK(ok)) {
            LOG_MAGICK_ERR(wand);
            DestroyMagickWand(wand);
            MagickWandTerminus();
            return NULL;
        }
        MagickWand * clone = NewMagickWand();
        PixelWand * pwand = NewPixelWand();
        if (clone == NULL || pwand == NULL) {
            LOG_MAGICK_ERR(wand);
            DestroyMagickWand(wand);
            MagickWandTerminus();
            return NULL;
        }
        
        ok = PixelSetColor(pwand, bgcolor);
        if (!IS_OK(ok)) {
            LOG_MAGICK_ERR(wand);
            DestroyMagickWand(wand);
            DestroyMagickWand(clone);
            MagickWandTerminus();
            return NULL;
        }
        ok = MagickNewImage(clone, fit_w, fit_h, pwand);
        if (!IS_OK(ok)) {
            LOG_MAGICK_ERR(wand);
            DestroyMagickWand(wand);
            DestroyMagickWand(clone);
            MagickWandTerminus();
            return NULL;
        }
        ssize_t x = (fit_w / 2) - (w / 2);
        ssize_t y = (fit_h / 2) - (h / 2);
        //ok = MagickCompositeImageGravity(clone, wand, CopyCompositeOp,
        //                                 CenterGravity);
        ok = MagickCompositeImage(clone, wand,CopyCompositeOp, x, y);
        if (!IS_OK(ok)) {
            LOG_MAGICK_ERR(wand);
            DestroyMagickWand(wand);
            DestroyMagickWand(clone);
            MagickWandTerminus();
            return NULL;
        }
        
        DestroyMagickWand(wand);
        wand = clone;
    }
    
    if (invert) {
        ok = MagickNegateImage(wand, 0);
        if (!IS_OK(ok)) {
            LOG_MAGICK_ERR(wand);
            DestroyMagickWand(wand);
            MagickWandTerminus();
            return NULL;
        }
    }
    
    double * pixels = malloc(fit_w * fit_h * pixel_size * sizeof(double));
    if (pixels == NULL) {
        fprintf(stderr, "Could not allocate memory for image pixels!\n");
        DestroyMagickWand(wand);
        MagickWandTerminus();
        return NULL;
    }
    
    ok = MagickExportImagePixels(wand, 0, 0, fit_w, fit_h, channel_map,
                                 DoublePixel, pixels);
    if (!IS_OK(ok)) {
        LOG_MAGICK_ERR(wand);
        free(pixels);
        pixels = NULL;
    }

    if (dump_file != NULL) {
        //dumpPixels(fit_w, fit_h, pixel_size, pixels);
        MagickWand * swand = NewMagickWand();
        if (swand != NULL) {
            ok = MagickConstituteImage(swand, fit_w, fit_h, channel_map,
                                       DoublePixel, pixels);
            if (!IS_OK(ok)) {
                LOG_MAGICK_ERR(swand);
                fprintf(stderr, "Failed to constitute image from pixels!\n");
            } else {
                ok = MagickWriteImage(swand, dump_file);
                if (!IS_OK(ok)) {
                    LOG_MAGICK_ERR(swand);
                    fprintf(stderr, "Failed to save dumped image!\n");
                }
            }
            DestroyMagickWand(swand);
        } else fprintf(stderr, "Failed to constitute image from pixels!\n");
    }
    
    DestroyMagickWand(wand);
    MagickWandTerminus();
    return pixels;
}

int PSClassifyImage(PSNeuralNetwork * network, char * filename, int grayscale,
                    int invert, char* bgcolor, char* dump_file)
{
    int input_size = network->input_size;
    int w = (int) sqrt((double) input_size);
    double * pixels = getImagePixels(filename, w, w, grayscale,
                                     invert, bgcolor, dump_file);
    if (pixels == NULL) {
        PSErr("PSClassifyImage", "Failed to get pixels from %s", filename);
        return -1;
    }
    printf("Feeding image: %s\n", filename);
    int res = PSClassify(network, pixels);
    free(pixels);
    return res;
}

