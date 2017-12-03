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

#ifndef __PS_TEST_H
#define __PS_TEST_H

#define NOT_PERFORMED -1

typedef int (* TestFunction) (void* test_case, void* test);
typedef int (* SetupFunction) (void* test_case);
typedef int (* TeardownFunction) (void* test_case);

typedef struct {
    char * name;
    char * error_message;
    int status;
    TestFunction run;
} Test;

typedef struct {
    char * name;
    SetupFunction setup;
    TeardownFunction teardown;
    int count;
    Test * tests;
    void ** data;
} TestCase;

TestCase * createTest(char * name);
Test * addTest(TestCase * test_case, char * name, char * errmsg,
               TestFunction func);
int performTests(TestCase * test_case);
void deleteTest(TestCase * test_case);

#endif // __PS_TEST_H
