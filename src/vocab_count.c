//  Tool to extract unigram counts
//
//  GloVe: Global Vectors for Word Representation
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
 *  vocab_count.c 是4个核心文件中的第一个文件，在demo.sh中给出的使用样例是：
 *   $ build/vocab_count -min-count 5 -verbose 2 < text8 > vocab.txt
 *  共有3个参数：
 *   - verbose:   控制是否输出debug信息
 *   - min_count: 最小词频，低于这个阈值的词会被丢弃
 *   - max_vocab: 最大词数，总词数不超过这个阈值，如果超过了，会有比min_count高的词被丢弃
 *  程序使用标准输入和输出，输入是空格分隔的文件，输出是词频统计
 *  
 *  程序执行时，先遍历整个文件，读取每一个词，并使用哈希表来存储词频；读完后把哈希表转成数组。
 *  哈希表用链表的方式解决哈希冲突，并且使用了类似LRU的策略来提速。
 *  
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 每个单词最长长度，如果超出会被截断成2个
#define MAX_STRING_LENGTH 1000
// 哈希表的大小 2^20
#define TSIZE	1048576
// 哈希函数中使用
#define SEED	1159241
// 哈希函数，实际本程序中只有这一个哈希函数
#define HASHFN  bitwisehash

// 最终词表的数组的节点
typedef struct vocabulary {
    // 词
    char *word;
    // 词频
    long long count;
} VOCAB;

// 每一条哈希记录的结构，同时也是单向链表节点
typedef struct hashrec {
    // 词
    char *word;
    // 统计词频
    long long count;
    // 链表，用来解决哈希碰撞
    struct hashrec *next;
} HASHREC;

// 控制是否输出debug信息
int verbose = 2; // 0, 1, or 2
// 最小词频，低于这个阈值的词会被丢弃
long long min_count = 1; // min occurrences for inclusion in vocab
// 最大词数，总词数不超过这个阈值，如果超过了，会有比min_count高的词被丢弃
long long max_vocab = 0; // max_vocab = 0 for no limit


/* Efficient string comparison */
// 快速比较两个词是不是相同
int scmp( char *s1, char *s2 ) {
    while (*s1 != '\0' && *s1 == *s2) {s1++; s2++;}
    return(*s1 - *s2);
}


/* Vocab frequency comparison; break ties alphabetically */
// 比较两个词的词频，a>b返回-1；a<b返回1；a=b时返回scmp的值（不为0的某个整数）
int CompareVocabTie(const void *a, const void *b) {
    long long c;
    if ( (c = ((VOCAB *) b)->count - ((VOCAB *) a)->count) != 0) return ( c > 0 ? 1 : -1 );
    else return (scmp(((VOCAB *) a)->word,((VOCAB *) b)->word));
    
}

/* Vocab frequency comparison; no tie-breaker */
// tie-breaker : 平局决胜
// 比较两个词的词频，a>b返回-1；a<b返回1；a=b返回0，不继续比较
int CompareVocab(const void *a, const void *b) {
    long long c;
    if ( (c = ((VOCAB *) b)->count - ((VOCAB *) a)->count) != 0) return ( c > 0 ? 1 : -1 );
    else return 0;
}

/* Move-to-front hashing and hash function from Hugh Williams, http://www.seg.rmit.edu.au/code/zwh-ipl/ */

/* Simple bitwise hash function */
// 哈希函数
unsigned int bitwisehash(char *word, int tsize, unsigned int seed) {
    char c;
    unsigned int h;
    h = seed;
    for (; (c =* word) != '\0'; word++) h ^= ((h << 5) + c + (h >> 2));
    return((unsigned int)((h&0x7fffffff) % tsize));
}

/* Create hash table, initialise pointers to NULL */
// 新建一个大小为TSIZE的哈希表，用链表解决哈希冲突
HASHREC ** inithashtable() {
    int	i;
    HASHREC **ht;
    ht = (HASHREC **) malloc( sizeof(HASHREC *) * TSIZE );
    for (i = 0; i < TSIZE; i++) ht[i] = (HASHREC *) NULL;
    return(ht);
}

/* Search hash table for given string, insert if not found */
// 搜索一个词是否在哈希表中，没有的话就创建
// 本代码中使用链表来解决哈希冲突
void hashinsert(HASHREC **ht, char *w) {
    // 指向一条哈希表记录的指针
    HASHREC	*htmp, *hprv;
    // 计算这个词的哈希值，默认调用bitwisehash函数
    unsigned int hval = HASHFN(w, TSIZE, SEED);
    
    // 通过哈希值寻找对应的哈希表记录，如果发现非空并且不是当前词，
    // 说明遇到哈希冲突，则循环到链表的下一个节点，直到找到一个空记录
    for (hprv = NULL, htmp = ht[hval]; htmp != NULL && scmp(htmp->word, w) != 0; hprv = htmp, htmp = htmp->next);
    // 如果当前词不存在，创建
    if (htmp == NULL) {
        // 创建一条新的哈希表记录
        htmp = (HASHREC *) malloc( sizeof(HASHREC) );
        // 把词赋值
        htmp->word = (char *) malloc( strlen(w) + 1 );
        strcpy(htmp->word, w);
        // 词频初始化为1
        htmp->count = 1;
        htmp->next = NULL;
        // 指针操作，如果有哈希冲突，要跟前面的节点连起来，否则直接赋值到哈希表
        if ( hprv==NULL )
            ht[hval] = htmp;
        else
            hprv->next = htmp;
    }
    else {
        /* new records are not moved to front */
        // 如果当前词已存在，把词频加1
        htmp->count++;
        // 如果当前词不是链表的头部，即有哈希冲突，则把当前词移到链表的头部
        // 这是一个类似LRU的策略，十分有效，因为哈希之后链表里的词互相有关的概率很小
        // 而在一段话中一个词反复出现的概率又很高，所以这样做能节约很多遍历链表的操作
        if (hprv != NULL) {
            /* move to front on access */
            hprv->next = htmp->next;
            htmp->next = ht[hval];
            ht[hval] = htmp;
        }
    }
    return;
}

// 统计词频，程序最主要的逻辑
int get_counts() {
    // i、j是循环变量，vocab_size
    long long i = 0, j = 0, vocab_size = 12500;
    // scanf用的format参数
    char format[20];
    // 用来存每次读入的词
    char str[MAX_STRING_LENGTH + 1];
    // 创建哈希表
    HASHREC **vocab_hash = inithashtable();
    // 临时变量，一条哈希表记录的指针
    HASHREC *htmp;
    // 一个词
    VOCAB *vocab;
    // 输入句柄，直接使用了标准输入
    FILE *fid = stdin;
    
    fprintf(stderr, "BUILDING VOCABULARY\n");
    if (verbose > 1) fprintf(stderr, "Processed %lld tokens.", i);
    // 创建format，每次读取限定最长长度不超过MAX_STRING_LENGTH的字符串，默认为1000
    sprintf(format,"%%%ds",MAX_STRING_LENGTH);
    // 每次从标准输入中读取一个词，直到文件结束
    while (fscanf(fid, format, str) != EOF) { // Insert all tokens into hashtable
        // <unk> 是系统默认项，用户文件中不许出现，直接退出程序
        if (strcmp(str, "<unk>") == 0) {
            fprintf(stderr, "\nError, <unk> vector found in corpus.\nPlease remove <unk>s from your corpus (e.g. cat text8 | sed -e 's/<unk>/<raw_unk>/g' > text8.new)");
            return 1;
        }
        // 插入哈希表
        hashinsert(vocab_hash, str);
        // 统计总token数，重复词重复计入
        if (((++i)%100000) == 0) if (verbose > 1) fprintf(stderr,"\033[11G%lld tokens.", i);
    }
    // 这个i是整个语料库中的token的总数，同一个词重复出现会重复计算，不是词表中词的数目
    if (verbose > 1) fprintf(stderr, "\033[0GProcessed %lld tokens.\n", i);
    // 创建词表，vocab_size为默认值，如果超了会realloc，奇怪vocab_size怎么没有设成2的幂次
    vocab = malloc(sizeof(VOCAB) * vocab_size);
    // 遍历整个哈希表，TSIZE是哈希表的大小
    for (i = 0; i < TSIZE; i++) { // Migrate vocab to array
        htmp = vocab_hash[i];
        // 将一条哈希表项，以及可能存在的整条链表，都存到词表中
        while (htmp != NULL) {
            vocab[j].word = htmp->word;
            vocab[j].count = htmp->count;
            j++;
            // 如果超了词表的大小，realloc一下
            if (j>=vocab_size) {
                vocab_size += 2500;
                vocab = (VOCAB *)realloc(vocab, sizeof(VOCAB) * vocab_size);
            }
            htmp = htmp->next;
        }
    }
    if (verbose > 1) fprintf(stderr, "Counted %lld unique words.\n", j);
    // 不太喜欢这种压缩的if for while的结构，用一下大括号会死啊...
    // 如果用户设定了max_vocab，并且总词数超过了大小max_vocab限制的大小
    // 那么先不使用平局决胜地排一次序，这样尾部频率相同的词就是乱序的，截尾去掉其中一部分的时候，就近似随机
    if (max_vocab > 0 && max_vocab < j)
        // If the vocabulary exceeds limit, first sort full vocab by frequency without alphabetical tie-breaks.
        // This results in pseudo-random ordering for words with same frequency, so that when truncated, the words span whole alphabet
        qsort(vocab, j, sizeof(VOCAB), CompareVocab);
    // 如果max_vocab没有设置（默认为0），或者比总词数j要大，则把max_vocab设成总词数
    else max_vocab = j;
    // 排序（或者是重新排序），按词频排序，词频相同时按字典序排序
    // 实际上如果是重新排序，多写几行代码可以比较遍历整个数组，即从尾部找出所以相同频率的词。不过看起来作者并不在乎。
    qsort(vocab, max_vocab, sizeof(VOCAB), CompareVocabTie); //After (possibly) truncating, sort (possibly again), breaking ties alphabetically
    
    // 遍历整个词表并输出
    for (i = 0; i < max_vocab; i++) {
        // 其实这个完全可以在前面哈希表转词表数组的时候过滤，可以节约很多内存，因为词频分布是非常长尾的
        if (vocab[i].count < min_count) { // If a minimum frequency cutoff exists, truncate vocabulary
            if (verbose > 0) fprintf(stderr, "Truncating vocabulary at min count %lld.\n",min_count);
            break;
        }
        // 直接输出结果到标准输出
        printf("%s %lld\n",vocab[i].word,vocab[i].count);
    }
    
    // 输出两个信息
    if (i == max_vocab && max_vocab < j) if (verbose > 0) fprintf(stderr, "Truncating vocabulary at size %lld.\n", max_vocab);
    fprintf(stderr, "Using vocabulary of size %lld.\n\n", i);
    return 0;
}

// 查找是否有某个参数，机智地复用了scmp
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

// 入口函数
int main(int argc, char **argv) {
    int i;
    // 如果不带参数，则输出参数介绍并退出程序
    if (argc == 1) {
        printf("Simple tool to extract unigram counts\n");
        printf("Author: Jeffrey Pennington (jpennin@stanford.edu)\n\n");
        printf("Usage options:\n");
        printf("\t-verbose <int>\n");
        printf("\t\tSet verbosity: 0, 1, or 2 (default)\n");
        printf("\t-max-vocab <int>\n");
        // 此处应有换行
        printf("\t\tUpper bound on vocabulary size, i.e. keep the <int> most frequent words. The minimum frequency words are randomly sampled so as to obtain an even distribution over the alphabet.\n");
        printf("\t-min-count <int>\n");
        printf("\t\tLower limit such that words which occur fewer than <int> times are discarded.\n");
        printf("\nExample usage:\n");
        printf("./vocab_count -verbose 2 -max-vocab 100000 -min-count 10 < corpus.txt > vocab.txt\n");
        return 0;
    }
    
    // 控制是否输出debug信息
    if ((i = find_arg((char *)"-verbose", argc, argv)) > 0) verbose = atoi(argv[i + 1]);
    // 最大词数，总词数不超过这个阈值，如果超过了，会有比min_count高的词被丢弃
    if ((i = find_arg((char *)"-max-vocab", argc, argv)) > 0) max_vocab = atoll(argv[i + 1]);
    // 最小词频，低于这个阈值的词会被丢弃
    if ((i = find_arg((char *)"-min-count", argc, argv)) > 0) min_count = atoll(argv[i + 1]);
    // 统计词频
    return get_counts();
}

