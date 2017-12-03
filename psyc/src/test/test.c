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
#include <unistd.h>
#include <time.h>
#include "test.h"

#define RED     "\x1b[31m"
#define GREEN   "\x1b[32m"
#define YELLOW  "\x1b[33m"
#define BLUE    "\x1b[34m"
#define MAGENTA "\x1b[35m"
#define CYAN    "\x1b[36m"
#define WHITE   "\x1b[37m"
#define BOLD    "\x1b[1m"
#define DIM     "\x1b[2m"
#define HIDDEN  "\x1b[8m"
#define RESET   "\x1b[0m"

int stdout_fd = -999;

TestCase * createTest(char * name) {
    TestCase * test_case = malloc(sizeof(TestCase));
    test_case->name = name;
    test_case->setup = NULL;
    test_case->teardown = NULL;
    test_case->count = 0;
    test_case->tests = NULL;
    test_case->data = NULL;
    return test_case;
}

Test * addTest(TestCase * test_case, char * name, char * errmsg,
               TestFunction func) {
    Test test;
    test.name = name;
    test.error_message = errmsg;
    test.run = func;
    test.status = NOT_PERFORMED;
    test_case->count++;
    if (test_case->tests == NULL) {
        test_case->tests = malloc(sizeof(Test));
    } else {
        test_case->tests = realloc(test_case->tests,
                                   sizeof(Test) * test_case->count);
    }
    test_case->tests[test_case->count - 1] = test;
    return &(test_case->tests[test_case->count - 1]);
}


int performTests(TestCase * test_case) {
    printf("\n");
    printf(BOLD "Performing tests on %s\n", test_case->name);
    printf(RESET);
    if (test_case->setup != NULL) {
        printf(" -> setup\n");
        printf(DIM);
        int ok = test_case->setup(test_case);
        printf(RESET);
        if (!ok) {
            printf(RED "Setup failed!\n" RESET);
            return 1;
        }
    }
    int i, errors = 0, count = test_case->count;
    time_t start_t, end_t;
    time(&start_t);
    for (i = 0; i < count; i++) {
        Test * test = &(test_case->tests[i]);
        printf(" -> [%d] ", i);
        printf(CYAN "%s", test->name); printf(":");
        printf(RESET);
        //printf(HIDDEN);
        stdout_fd = dup(fileno(stdout));
        freopen("/dev/null", "w", stdout);
        test->status = test->run(test_case, test);
        fflush(stdout);
        fclose(stdout);
        stdout = fdopen(stdout_fd, "w");
        //printf(RESET);
        if (!test->status) {
            printf(RED "\tFAILED");
            if (test->error_message != NULL)
                printf(YELLOW "\n      %s\n", test->error_message);
            errors++;
        } else printf(GREEN "\tOK");
        printf("\n" RESET);
    }
    if (test_case->teardown != NULL) {
        printf(" -> teardown\n");
        int ok = test_case->teardown(test_case);
        if (!ok) {
            printf(RED "Teardown failed!\n" RESET);
            return 1;
        }
    }
    time(&end_t);
    printf("Tests performed in %d sec.\n", (int) (end_t - start_t));
    printf("Found ");
    if (errors > 0) printf(RED "%d errors.\n", errors);
    else printf(GREEN "no errors.\n");
    printf(RESET);
    return errors;
}

void deleteTest(TestCase * test_case) {
    if (test_case->data != NULL) free(test_case->data);
    if (test_case->tests != NULL) {
        int i;
        for (i = 0; i < test_case->count; i++) {
            Test * test = &(test_case->tests[i]);
            if (test->error_message != NULL) free(test->error_message);
        }
        free(test_case->tests);
    }
    free(test_case);
}
