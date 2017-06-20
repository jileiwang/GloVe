//  Tool to shuffle entries of word-word cooccurrence files
//
//  Copyright (c) 2014 The Board of Trustees of
//  The Leland Stanford Junior University. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//
//  For more information, bug reports, fixes, contact:
//    Jeffrey Pennington (jpennin@stanford.edu)
//    GlobalVectors@googlegroups.com
//    http://nlp.stanford.edu/projects/glove/



/***
 *  
 *  shuffle.c 是4个核心文件中的第三个文件，在demo.sh中给出的使用样例是：
 *  $ build/shuffle -memory 4.0 -verbose 2 < cooccurrence.bin > cooccurrence.shuf.bin
 *
 *  本程序的目的是把第2个程序cooccur输出的内容打散，来保证之后训练的随机性，因为glove中用的应该是随机梯度下降
 *  输入是有序的，可能会很大而没法一次性装载到内存，所以需要分次打散
 *  打散的方法是每次读取一个大小为array_size条记录的chuck，打散整个chuck并存到一个临时文件
 *  读取完全部的输入并得到num个临时文件后，开始每次从每个临时文件中读取array_size/num条记录，组成一个新的chuck，
 *  再次打散，并输出到标准输出，重复这样num次就可以把所有的文件输出完毕
 *  可以证明，只要每次打散chuck是纯随机的，那么最终输出的结果也是纯随机的，输入的每一条记录等概率出现在输出的每一个位置
 *
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAX_STRING_LENGTH 1000

static const long LRAND_MAX = ((long) RAND_MAX + 2) * (long)RAND_MAX;
typedef double real;

// 共现记录，word1和word2是词在vocab_count词频统计的输出文件中的序号，序号越小词频越大
typedef struct cooccur_rec {
    int word1;
    int word2;
    // 加权后的共现次数
    real val;
} CREC;

// 用来控制输出log
int verbose = 2; // 0, 1, or 2
// 数组长度，这个数组的大小基本就是使用的内存的大小，也是每个临时文件的大小
long long array_size = 2000000; // size of chunks to shuffle individually
// 临时文件的文件头，默认是temp_shuffle，所以临时文件就是temp_shuffle_0000.bin的格式命名
char *file_head; // temporary file string
// 限制内存大小，这是一个粗略的限制，默认是2G
real memory_limit = 2.0; // soft limit, in gigabytes

/* Efficient string comparison */
// 比较两个词书佛欧相同，这个程序中好像没有用到
int scmp( char *s1, char *s2 ) {
    while (*s1 != '\0' && *s1 == *s2) {s1++; s2++;}
    return(*s1 - *s2);
}


/* Generate uniformly distributed random long ints */
// 生成一个随机数
static long rand_long(long n) {
    long limit = LRAND_MAX - LRAND_MAX % n;
    long rnd;
    do {
        rnd = ((long)RAND_MAX + 1) * (long)rand() + (long)rand();
    } while (rnd >= limit);
    return rnd % n;
}

/* Write contents of array to binary file */
// 把整个数组中的记录，写到输出文件或标准输出
int write_chunk(CREC *array, long size, FILE *fout) {
    long i = 0;
    for (i = 0; i < size; i++) fwrite(&array[i], sizeof(CREC), 1, fout);
    return 0;
}

/* Fisher-Yates shuffle */
// 打乱数组
void shuffle(CREC *array, long n) {
    long i, j;
    CREC tmp;
    // 对每一个数，随机交换一次
    for (i = n - 1; i > 0; i--) {
        j = rand_long(i + 1);
        tmp = array[j];
        array[j] = array[i];
        array[i] = tmp;
    }
}

/* Merge shuffled temporary files; doesn't necessarily produce a perfect shuffle, but good enough */
int shuffle_merge(int num) {
    long i, j, k, l = 0;
    int fidcounter = 0;
    CREC *array;
    char filename[MAX_STRING_LENGTH];
    FILE **fid, *fout = stdout;
    
    // 重新开辟一个数组，其实复用上一个也可以吧
    array = malloc(sizeof(CREC) * array_size);
    // 这个数组用来存各个文件的句柄
    fid = malloc(sizeof(FILE) * num);
    // 一共有num个临时文件
    for (fidcounter = 0; fidcounter < num; fidcounter++) { //num = number of temporary files to merge
        sprintf(filename,"%s_%04d.bin",file_head, fidcounter);
        // 打开文件
        fid[fidcounter] = fopen(filename, "rb");
        // 如果失败就退出
        if (fid[fidcounter] == NULL) {
            fprintf(stderr, "Unable to open file %s.\n",filename);
            return 1;
        }
    }
    if (verbose > 0) fprintf(stderr, "Merging temp files: processed %ld lines.", l);
    
    // 直到所有文件都读完
    while (1) { //Loop until EOF in all files
        // i用来记录一次while循环中读取的记录的数目
        i = 0;
        //Read at most array_size values into array, roughly array_size/num from each temp file
        // 从每个文件中读取array_size / num条记录到数组
        for (j = 0; j < num; j++) {
            // 如果当前文件空了，直接读下一个
            if (feof(fid[j])) continue;
            // 从当前文件中读取array_size / num条记录到数组
            for (k = 0; k < array_size / num; k++){
                // 读取一条记录
                fread(&array[i], sizeof(CREC), 1, fid[j]);
                // 如果EOF了，退出当前循环，读取下一个文件
                if (feof(fid[j])) break;
                // 记录读取记录的数目
                i++;
            }
        }
        // 如果一条都没读到，说明所有文件都EOF了，退出
        if (i == 0) break;
        // 记录一共读取了多少条记录
        l += i;
        // 打乱数组
        shuffle(array, i-1); // Shuffles lines between temp files
        // 把打乱后的数组写到标准输出
        write_chunk(array,i,fout);
        if (verbose > 0) fprintf(stderr, "\033[31G%ld lines.", l);
    }
    fprintf(stderr, "\033[0GMerging temp files: processed %ld lines.", l);
    // 关闭所有临时文件，并删除文件
    for (fidcounter = 0; fidcounter < num; fidcounter++) {
        fclose(fid[fidcounter]);
        sprintf(filename,"%s_%04d.bin",file_head, fidcounter);
        remove(filename);
    }
    fprintf(stderr, "\n\n");
    // 释放内存，程序结束
    free(array);
    return 0;
}

/* Shuffle large input stream by splitting into chunks */
// 通过把文件切成一个个chunk的方式来打乱
int shuffle_by_chunks() {
    long i = 0, l = 0;
    int fidcounter = 0;
    char filename[MAX_STRING_LENGTH];
    CREC *array;
    FILE *fin = stdin, *fid;
    array = malloc(sizeof(CREC) * array_size);
    
    fprintf(stderr,"SHUFFLING COOCCURRENCES\n");
    if (verbose > 0) fprintf(stderr,"array size: %lld\n", array_size);
    // 打开第一个文件
    sprintf(filename,"%s_%04d.bin",file_head, fidcounter);
    fid = fopen(filename,"w");
    // 如果失败就退出
    if (fid == NULL) {
        fprintf(stderr, "Unable to open file %s.\n",filename);
        return 1;
    }
    if (verbose > 1) fprintf(stderr, "Shuffling by chunks: processed 0 lines.");
    
    // 循环直到读完所有文件中的记录
    while (1) { //Continue until EOF
        // i用来记录目前数组中已经存的记录的个数
        // 如果要超过数组大小了，就输出到文件
        if (i >= array_size) {// If array is full, shuffle it and save to temporary file
            // 打乱一下
            shuffle(array, i-2);
            // l记录了已经读取了多少条记录
            l += i;
            if (verbose > 1) fprintf(stderr, "\033[22Gprocessed %ld lines.", l);
            // 把这个chuck，也就是一个数组，写入一个临时文件
            write_chunk(array,i,fid);
            // 关闭文件
            fclose(fid);
            // 用来存储一共写了多少个临时文件了
            fidcounter++;
            // 打开下一个临时文件
            sprintf(filename,"%s_%04d.bin",file_head, fidcounter);
            fid = fopen(filename,"w");
            // 如果打开失败 就退出
            if (fid == NULL) {
                fprintf(stderr, "Unable to open file %s.\n",filename);
                return 1;
            }
            // 数组计数器归0
            i = 0;
        }
        // 读取下一条记录，存到数组的
        fread(&array[i], sizeof(CREC), 1, fin);
        if (feof(fin)) break;
        i++;
    }
    // 打乱最后一个chuck
    shuffle(array, i-2); //Last chunk may be smaller than array_size
    // 写入最后一个chuck
    write_chunk(array,i,fid);
    // 记录一个写了多少个记录
    l += i;
    if (verbose > 1) fprintf(stderr, "\033[22Gprocessed %ld lines.\n", l);
    if (verbose > 1) fprintf(stderr, "Wrote %d temporary file(s).\n", fidcounter + 1);
    // 关闭文件 清理内存
    fclose(fid);
    free(array);
    // 还没完呢！把所有文件重新合并成一个文件，这个过程中也会打乱的
    return shuffle_merge(fidcounter + 1); // Merge and shuffle together temporary files
}

// 用来查找某一个参数是否指定以及其位置
int find_arg(char *str, int argc, char **argv) {
    int i;
    for (i = 1; i < argc; i++) {
        if (!scmp(str, argv[i])) {
            if (i == argc - 1) {
                printf("No argument given for %s\n", str);
                exit(1);
            }
            return i;
        }
    }
    return -1;
}

// 入口函数，读取参数，进入程序主逻辑
int main(int argc, char **argv) {
    int i;
    file_head = malloc(sizeof(char) * MAX_STRING_LENGTH);
    
    if (argc == 1) {
        printf("Tool to shuffle entries of word-word cooccurrence files\n");
        printf("Author: Jeffrey Pennington (jpennin@stanford.edu)\n\n");
        printf("Usage options:\n");
        printf("\t-verbose <int>\n");
        printf("\t\tSet verbosity: 0, 1, or 2 (default)\n");
        printf("\t-memory <float>\n");
        printf("\t\tSoft limit for memory consumption, in GB; default 4.0\n");
        printf("\t-array-size <int>\n");
        printf("\t\tLimit to length <int> the buffer which stores chunks of data to shuffle before writing to disk. \n\t\tThis value overrides that which is automatically produced by '-memory'.\n");
        printf("\t-temp-file <file>\n");
        printf("\t\tFilename, excluding extension, for temporary files; default temp_shuffle\n");
        
        printf("\nExample usage: (assuming 'cooccurrence.bin' has been produced by 'coccur')\n");
        printf("./shuffle -verbose 2 -memory 8.0 < cooccurrence.bin > cooccurrence.shuf.bin\n");
        return 0;
    }
   
    // 用来控制输出log
    if ((i = find_arg((char *)"-verbose", argc, argv)) > 0) verbose = atoi(argv[i + 1]);
    // 临时文件的文件头，默认是temp_shuffle，所以临时文件就是temp_shuffle_0000.bin的格式命名
    if ((i = find_arg((char *)"-temp-file", argc, argv)) > 0) strcpy(file_head, argv[i + 1]);
    else strcpy(file_head, (char *)"temp_shuffle");
    // 限制内存大小，这是一个粗略的限制，默认是4G
    if ((i = find_arg((char *)"-memory", argc, argv)) > 0) memory_limit = atof(argv[i + 1]);
    // 通过内存大小，来计算要开辟的数组的长度
    array_size = (long long) (0.95 * (real)memory_limit * 1073741824/(sizeof(CREC)));
    // 如果用户指定了数组长度，那么直接赋值，上面的计算就是娱乐而已了
    if ((i = find_arg((char *)"-array-size", argc, argv)) > 0) array_size = atoll(argv[i + 1]);
    // 开始打乱
    return shuffle_by_chunks();
}

