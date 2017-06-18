//  Tool to calculate word-word cooccurrence statistics
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
 *  vocab_count.c 是4个核心文件中的第二个文件，用于构造所有词的共现矩阵（Co-occurence Matrix）
 *  在demo.sh中给出的使用样例是：
 *   $ build/cooccur -memory 4.0 -vocab-file vocab.txt -verbose 2 -window-size 15 < text8 > cooccurrence.bin
 *  共有3个参数：
 *   - verbose:   控制是否输出debug信息
 *   - symmetric: 1
 *   - window-size  15
 *   - vocab-file  vocab.txt
 *   - memory
 *   - max-product
 *   - overflow-length
 *   - overflow-file
 *   - 

 *  程序使用标准输入和输出，输入是空格分隔的文件，输出是词频统计
 *  
 *  程序执行时，先遍历整个文件，读取每一个词，并使用哈希表来存储词频；读完后把哈希表转成数组。
 *  哈希表用链表的方式解决哈希冲突，并且使用了类似LRU的策略来提速。




 *  
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TSIZE 1048576
#define SEED 1159241
#define HASHFN bitwisehash

static const int MAX_STRING_LENGTH = 1000;
typedef double real;

// Cooccurence Record, 两个词共现的值
typedef struct cooccur_rec {
    int word1;
    int word2;
    real val;
} CREC;

// Cooccurence Record ID, 两个词共现的值以及id
typedef struct cooccur_rec_id {
    int word1;
    int word2;
    real val;
    int id;
} CRECID;

// 词频表的哈希表项
typedef struct hashrec {
    char	*word;
    long long id;
    struct hashrec *next;
} HASHREC;

int verbose = 2; // 0, 1, or 2
long long max_product; // Cutoff for product of word frequency ranks below which cooccurrence counts will be stored in a compressed full array
long long overflow_length; // Number of cooccurrence records whose product exceeds max_product to store in memory before writing to disk
int window_size = 15; // default context window size
int symmetric = 1; // 0: asymmetric, 1: symmetric
real memory_limit = 3; // soft limit, in gigabytes, used to estimate optimal array sizes
char *vocab_file, *file_head;

/* Efficient string comparison */
// 比较两个词是否相同
int scmp( char *s1, char *s2 ) {
    while (*s1 != '\0' && *s1 == *s2) {s1++; s2++;}
    return(*s1 - *s2);
}

/* Move-to-front hashing and hash function from Hugh Williams, http://www.seg.rmit.edu.au/code/zwh-ipl/ */

/* Simple bitwise hash function */
// 哈希函数，使用了位运算
unsigned int bitwisehash(char *word, int tsize, unsigned int seed) {
    char c;
    unsigned int h;
    h = seed;
    for (; (c =* word) != '\0'; word++) h ^= ((h << 5) + c + (h >> 2));
    return((unsigned int)((h&0x7fffffff) % tsize));
}

/* Create hash table, initialise pointers to NULL */
// 创建一个大小为TSIZE的空哈希表
HASHREC ** inithashtable() {
    int	i;
    HASHREC **ht;
    ht = (HASHREC **) malloc( sizeof(HASHREC *) * TSIZE );
    for (i = 0; i < TSIZE; i++) ht[i] = (HASHREC *) NULL;
    return(ht);
}

/* Search hash table for given string, return record if found, else NULL */
// 从哈希表中查找某个词 如果没有的话返回NULL
HASHREC *hashsearch(HASHREC **ht, char *w) {
    HASHREC	*htmp, *hprv;
    // 首先计算哈希值
    unsigned int hval = HASHFN(w, TSIZE, SEED);
    // 循环遍历哈希值位置的链表，找到对应词或者得到NULL
    for (hprv = NULL, htmp=ht[hval]; htmp != NULL && scmp(htmp->word, w) != 0; hprv = htmp, htmp = htmp->next);
    // 类似LRU的策略，找到词以后放到链表头部，方便下次查询
    if ( htmp != NULL && hprv!=NULL ) { // move to front on access
        hprv->next = htmp->next;
        htmp->next = ht[hval];
        ht[hval] = htmp;
    }
    return(htmp);
}

/* Insert string in hash table, check for duplicates which should be absent */
// 把一个词和其词频插入哈希表，同时还有其id。与count_vocab.c一样，使用链表解决哈希冲突
void hashinsert(HASHREC **ht, char *w, long long id) {
    HASHREC	*htmp, *hprv;
    unsigned int hval = HASHFN(w, TSIZE, SEED);
    // 查找哈希值对应的位置，如果已经存在则遍历使指针指到链表的末尾节点
    for (hprv = NULL, htmp = ht[hval]; htmp != NULL && scmp(htmp->word, w) != 0; hprv = htmp, htmp = htmp->next);
    if (htmp == NULL) {
        htmp = (HASHREC *) malloc(sizeof(HASHREC));
        htmp->word = (char *) malloc(strlen(w) + 1);
        strcpy(htmp->word, w);
        htmp->id = id;
        htmp->next = NULL;
        // 哈希值对应位置为空，直接插入该位置
        if (hprv == NULL) ht[hval] = htmp;
        // 已经存在节点，则插到链表末尾
        else hprv->next = htmp;
    }
    // 不应该存在同一个词插两次的情况
    else fprintf(stderr, "Error, duplicate entry located: %s.\n",htmp->word);
    return;
}

/* Read word from input stream */
// 从输入流中读入一个词
int get_word(char *word, FILE *fin) {
    int i = 0, ch;
    while (!feof(fin)) {
        ch = fgetc(fin);
        // 忽略回车符，这是因为考虑到windows文件的缘故吗
        if (ch == 13) continue;
        // 遇到空格Tab换行
        if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
            // 如果i>0，说明已经读取了一个词了
            if (i > 0) {
                // 如果是回车，要把回车放回去
                if (ch == '\n') ungetc(ch, fin);
                // 跳出循环，到程序末尾，返回值
                break;
            }
            // 此处是i==0的情况，即还没有读到词就读到了空格Tab换行
            // 如果是换行，直接返回1
            if (ch == '\n') return 1;
            // 空格或者Tab，则跳过继续查找词
            else continue;
        }
        // 正常字符，存入word中
        word[i++] = ch;
        // 如果一个词的长度超过了MAX_STRING_LENGTH，超出部分会被忽略
        if (i >= MAX_STRING_LENGTH - 1) i--;   // truncate words that exceed max length
    }
    // 结束字符串并返回
    word[i] = 0;
    return 0;
}

/* Write sorted chunk of cooccurrence records to file, accumulating duplicate entries */
// 把一大块的共现记录数组写到文件中，由于是排好序的，所以可以把两个词相同且顺序相同的记录合并
int write_chunk(CREC *cr, long long length, FILE *fout) {
    if (length == 0) return 0;

    long long a = 0;
    CREC old = cr[a];
    
    for (a = 1; a < length; a++) {
        // 词的顺序也必须一样
        if (cr[a].word1 == old.word1 && cr[a].word2 == old.word2) {
            old.val += cr[a].val;
            continue;
        }
        // 由于CREC类型是{int, int, real}，所以可以直接写入文件
        fwrite(&old, sizeof(CREC), 1, fout);
        old = cr[a];
    }
    fwrite(&old, sizeof(CREC), 1, fout);
    return 0;
}

/* Check if two cooccurrence records are for the same two words, used for qsort */
// 比较两条共现记录是不是同样的两个词，并且顺序相同。用于qsort排序
int compare_crec(const void *a, const void *b) {
    int c;
    if ( (c = ((CREC *) a)->word1 - ((CREC *) b)->word1) != 0) return c;
    else return (((CREC *) a)->word2 - ((CREC *) b)->word2);
    
}

/* Check if two cooccurrence records are for the same two words */
int compare_crecid(CRECID a, CRECID b) {
    int c;
    if ( (c = a.word1 - b.word1) != 0) return c;
    else return a.word2 - b.word2;
}

/* Swap two entries of priority queue */
void swap_entry(CRECID *pq, int i, int j) {
    CRECID temp = pq[i];
    pq[i] = pq[j];
    pq[j] = temp;
}

/* Insert entry into priority queue */
void insert(CRECID *pq, CRECID new, int size) {
    int j = size - 1, p;
    pq[j] = new;
    while ( (p=(j-1)/2) >= 0 ) {
        if (compare_crecid(pq[p],pq[j]) > 0) {swap_entry(pq,p,j); j = p;}
        else break;
    }
}

/* Delete entry from priority queue */
void delete(CRECID *pq, int size) {
    int j, p = 0;
    pq[p] = pq[size - 1];
    while ( (j = 2*p+1) < size - 1 ) {
        if (j == size - 2) {
            if (compare_crecid(pq[p],pq[j]) > 0) swap_entry(pq,p,j);
            return;
        }
        else {
            if (compare_crecid(pq[j], pq[j+1]) < 0) {
                if (compare_crecid(pq[p],pq[j]) > 0) {swap_entry(pq,p,j); p = j;}
                else return;
            }
            else {
                if (compare_crecid(pq[p],pq[j+1]) > 0) {swap_entry(pq,p,j+1); p = j + 1;}
                else return;
            }
        }
    }
}

/* Write top node of priority queue to file, accumulating duplicate entries */
int merge_write(CRECID new, CRECID *old, FILE *fout) {
    if (new.word1 == old->word1 && new.word2 == old->word2) {
        old->val += new.val;
        return 0; // Indicates duplicate entry
    }
    fwrite(old, sizeof(CREC), 1, fout);
    *old = new;
    return 1; // Actually wrote to file
}

/* Merge [num] sorted files of cooccurrence records */
int merge_files(int num) {
    int i, size;
    long long counter = 0;
    CRECID *pq, new, old;
    char filename[200];
    FILE **fid, *fout;
    fid = malloc(sizeof(FILE) * num);
    pq = malloc(sizeof(CRECID) * num);
    fout = stdout;
    if (verbose > 1) fprintf(stderr, "Merging cooccurrence files: processed 0 lines.");
    
    /* Open all files and add first entry of each to priority queue */
    for (i = 0; i < num; i++) {
        sprintf(filename,"%s_%04d.bin",file_head,i);
        fid[i] = fopen(filename,"rb");
        if (fid[i] == NULL) {fprintf(stderr, "Unable to open file %s.\n",filename); return 1;}
        fread(&new, sizeof(CREC), 1, fid[i]);
        new.id = i;
        insert(pq,new,i+1);
    }
    
    /* Pop top node, save it in old to see if the next entry is a duplicate */
    size = num;
    old = pq[0];
    i = pq[0].id;
    delete(pq, size);
    fread(&new, sizeof(CREC), 1, fid[i]);
    if (feof(fid[i])) size--;
    else {
        new.id = i;
        insert(pq, new, size);
    }
    
    /* Repeatedly pop top node and fill priority queue until files have reached EOF */
    while (size > 0) {
        counter += merge_write(pq[0], &old, fout); // Only count the lines written to file, not duplicates
        if ((counter%100000) == 0) if (verbose > 1) fprintf(stderr,"\033[39G%lld lines.",counter);
        i = pq[0].id;
        delete(pq, size);
        fread(&new, sizeof(CREC), 1, fid[i]);
        if (feof(fid[i])) size--;
        else {
            new.id = i;
            insert(pq, new, size);
        }
    }
    fwrite(&old, sizeof(CREC), 1, fout);
    fprintf(stderr,"\033[0GMerging cooccurrence files: processed %lld lines.\n",++counter);
    for (i=0;i<num;i++) {
        sprintf(filename,"%s_%04d.bin",file_head,i);
        remove(filename);
    }
    fprintf(stderr,"\n");
    return 0;
}

/* Collect word-word cooccurrence counts from input stream */
// 从标准输入中构造词-词共现矩阵
int get_cooccurrence() {
    // TODO
    int flag, x, y, fidcounter = 1;
    // TODO
    long long a, j = 0, k, id, counter = 0, ind = 0, vocab_size, w1, w2, *lookup, *history;
    // TODO
    char format[20], filename[200], str[MAX_STRING_LENGTH + 1];
    // TODO
    FILE *fid, *foverflow;
    // TODO
    real *bigram_table, r;
    // TODO
    HASHREC *htmp, **vocab_hash = inithashtable();
    // TODO
    CREC *cr = malloc(sizeof(CREC) * (overflow_length + 1));
    // TODO
    history = malloc(sizeof(long long) * window_size);
    
    // 输出参数信息
    fprintf(stderr, "COUNTING COOCCURRENCES\n");
    if (verbose > 0) {
        fprintf(stderr, "window size: %d\n", window_size);
        if (symmetric == 0) fprintf(stderr, "context: asymmetric\n");
        else fprintf(stderr, "context: symmetric\n");
    }
    if (verbose > 1) fprintf(stderr, "max product: %lld\n", max_product);
    if (verbose > 1) fprintf(stderr, "overflow length: %lld\n", overflow_length);
    // 读取词表文件用的格式，设定了最长词的限制
    sprintf(format,"%%%ds %%lld", MAX_STRING_LENGTH); // Format to read from vocab file, which has (irrelevant) frequency data
    if (verbose > 1) fprintf(stderr, "Reading vocab from file \"%s\"...", vocab_file);
    // 打开词表文件，失败就退出程序
    fid = fopen(vocab_file,"r");
    if (fid == NULL) {fprintf(stderr,"Unable to open vocab file %s.\n",vocab_file); return 1;}
    // 读取每一个词和其词频，插入哈希表
    while (fscanf(fid, format, str, &id) != EOF) hashinsert(vocab_hash, str, ++j); // Here id is not used: inserting vocab words into hash table with their frequency rank, j
    fclose(fid);
    // 获得词表大小
    vocab_size = j;
    j = 0;
    if (verbose > 1) fprintf(stderr, "loaded %lld words.\nBuilding lookup table...", vocab_size);
    
    /* Build auxiliary lookup table used to index into bigram_table */
    // lookup数组用来做bigram_table的下标索引
    // bigram_table是一个二维表，下标(x, y)满足x,y属于[1, vocab_size]且x*y < max_product
    // 索引方式为bigram_table(x, y) <- bigram_table[lookup[x-1] + y - 2]
    lookup = (long long *)calloc( vocab_size + 1, sizeof(long long) );
    if (lookup == NULL) {
        fprintf(stderr, "Couldn't allocate memory!");
        return 1;
    }
    lookup[0] = 1;
    // 由前面的分析，可以理解此处的代码：
    // 当a <= max_product / vocab_size时，下标(a,b)的b的取值范围是[1, vocab_size]，所以增量为vocab_size
    // 当a > max_product / vocab_size时，下标(a,b)的b的取值范围是[1, max_product/a]，所以增量为max_product/a
    for (a = 1; a <= vocab_size; a++) {
        if ((lookup[a] = max_product / a) < vocab_size) lookup[a] += lookup[a-1];
        else lookup[a] = lookup[a-1] + vocab_size;
    }
    if (verbose > 1) fprintf(stderr, "table contains %lld elements.\n",lookup[a-1]);
    
    /* Allocate memory for full array which will store all cooccurrence counts for words whose product of frequency ranks is less than max_product */
    // 开辟bigram_table的存储空间，用来存储所有w1*w2<max_product的部分的共现矩阵
    bigram_table = (real *)calloc( lookup[a-1] , sizeof(real) );
    if (bigram_table == NULL) {
        fprintf(stderr, "Couldn't allocate memory!");
        return 1;
    }
    
    fid = stdin;
    sprintf(format,"%%%ds",MAX_STRING_LENGTH);
    // file_head是overflow-file参数输入的，默认为overflow
    // fidcounter初始化为1
    sprintf(filename,"%s_%04d.bin",file_head, fidcounter);
    // 打开文件
    foverflow = fopen(filename,"w");
    if (verbose > 1) fprintf(stderr,"Processing token: 0");
    
    /* For each token in input stream, calculate a weighted cooccurrence sum within window_size */
    // 对标准输入里的每一个token，计算窗口内的加权的共现和
    while (1) {
        // ind初始化为0，如果overflow记录数的buffer快满了，就先写入临时文件
        if (ind >= overflow_length - window_size) { // If overflow buffer is (almost) full, sort it and write it to temporary file
            // cr是长度为overflow_length的CREC共现记录的数组
            qsort(cr, ind, sizeof(CREC), compare_crec);
            // 把这一组cr数组里的记录写入临时文件
            write_chunk(cr,ind,foverflow);
            // 关闭文件，每个文件都只写一次
            fclose(foverflow);
            // 文件数目计数器加1，打开下一个文件的句柄
            fidcounter++;
            sprintf(filename,"%s_%04d.bin",file_head,fidcounter);
            foverflow = fopen(filename,"w");
            // 数组内偏移量归零
            ind = 0;
        }
        // 取一个词，正常取词
        flag = get_word(str, fid);
        // 如果读完了就退出循环
        if (feof(fid)) break;
        // 如果是新的一行，重置j，进入下一次循环
        // 此处读取的换行符是上一行的末尾的换行符，所以下一次循环会从新一行的首字母读起
        // j表示一个词在一行中的位置是第j个词，不包括不在词表中的词
        if (flag == 1) {j = 0; continue;} // Newline, reset line index (j)
        // 统计输入的token的总数
        counter++;
        // 输出信息
        if ((counter%100000) == 0) if (verbose > 1) fprintf(stderr,"\033[19G%lld",counter);
        // 从哈希表中查找到这个词对应的哈希记录
        htmp = hashsearch(vocab_hash, str);
        // 如果不在哈希表中，进入下一轮循环
        if (htmp == NULL) continue; // Skip out-of-vocabulary words
        // 目标词的词频排名
        w2 = htmp->id; // Target word (frequency rank)
        for (k = j - 1; k >= ( (j > window_size) ? j - window_size : 0 ); k--) { // Iterate over all words to the left of target word, but not past beginning of line
            // history记录了各个之前读到的词的词频排名
            // 这个模窗口大小作为存储位置的实现，非常优雅，就是一个最简单的循环列表
            w1 = history[k % window_size]; // Context word (frequency rank)
            // w1是上下文词的词频排名，w2是当前词的词频排名，词频排名越大的词出现的次数越少
            // 当w1 * w2 < max_product时，说明两个词共现的概率比较高
            if ( w1 < max_product/w2 ) { // Product is small enough to store in a full array
                // lookup[w1-1] + w2 - 2相当于是一个矩阵下标(w1, w2)
                // 具体lookup函数的使用方法可以写篇短文介绍一下
                // 增量是加权的，权值是上下文词和当前词距离的反比
                bigram_table[lookup[w1-1] + w2 - 2] += 1.0/((real)(j-k)); // Weight by inverse of distance between words
                // 如果开启了对称性，则把w1和w2换过来也计算一下，相当于w1做当前词，w2做上下文词
                if (symmetric > 0) bigram_table[lookup[w2-1] + w1 - 2] += 1.0/((real)(j-k)); // If symmetric context is used, exchange roles of w2 and w1 (ie look at right context too)
            }
            else { // Product is too big, data is likely to be sparse. Store these entries in a temporary buffer to be sorted, merged (accumulated), and written to file when it gets full.
                // 如果w1 * w2太大，超过了max_product，概率上来说两个词共同出现的概率较小，很可能是非常稀疏的数据
                // 所以就存入cr数组，这是一个buffer数组，当满了以后会存入overflow的文件里，存入前会排序和合并相同项
                cr[ind].word1 = w1;
                cr[ind].word2 = w2;
                // 更新的量是距离的反比
                cr[ind].val = 1.0/((real)(j-k));
                // ind是cr数组存储的下标偏移
                ind++; // Keep track of how full temporary buffer is
                // 如果打开了symmetric对称性，那么w2也是w1的上下文，要用w2更新w1
                // 这也是个很漂亮的操作，可以避免了读到当前词后，往右继续读找当前词在右边的上下文
                // 而是等右边的词成为当前词后，再回过头来更新
                if (symmetric > 0) { // Symmetric context
                    cr[ind].word1 = w2;
                    cr[ind].word2 = w1;
                    cr[ind].val = 1.0/((real)(j-k));
                    ind++;
                }
            }
        }
        // 把当前词存入history，用来作为接下来读取的词的上下文
        history[j % window_size] = w2; // Target word is stored in circular buffer to become context word in the future
        // 下一个词在这句句子中的位置j
        j++;
    }
    
    /* Write out temp buffer for the final time (it may not be full) */
    // 最后一次存储cr数组中的记录，可能此时数组并不满
    if (verbose > 1) fprintf(stderr,"\033[0GProcessed %lld tokens.\n",counter);
    qsort(cr, ind, sizeof(CREC), compare_crec);
    write_chunk(cr,ind,foverflow);
    // bigram_table中的数据存入尾号0000的文件中
    sprintf(filename,"%s_0000.bin",file_head);
    
    /* Write out full bigram_table, skipping zeros */
    // 这一段代码是把bigram_table中的全部的非0数据存入文件中
    if (verbose > 1) fprintf(stderr, "Writing cooccurrences to disk");
    fid = fopen(filename,"w");
    j = 1e6;
    // 对1到vocab_size这全部的词进行遍历，x即word1
    for (x = 1; x <= vocab_size; x++) {
        if ( (long long) (0.75*log(vocab_size / x)) < j) {j = (long long) (0.75*log(vocab_size / x)); if (verbose > 1) fprintf(stderr,".");} // log's to make it look (sort of) pretty
        // 对word1对应的全部的word2进行遍历
        // 当x < max_product / vocab_size时，这个差值就是vocab_size
        // 否则是 max_product / x
        for (y = 1; y <= (lookup[x] - lookup[x-1]); y++) {
            // 如果记录不为0，写入文件，这个写入格式跟之前的格式一样
            if ((r = bigram_table[lookup[x-1] - 2 + y]) != 0) {
                // word1
                fwrite(&x, sizeof(int), 1, fid);
                // word2
                fwrite(&y, sizeof(int), 1, fid);
                // val
                fwrite(&r, sizeof(real), 1, fid);
            }
        }
    }
    
    // 关闭文件，释放各个存储空间
    if (verbose > 1) fprintf(stderr,"%d files in total.\n",fidcounter + 1);
    fclose(fid);
    fclose(foverflow);
    free(cr);
    free(lookup);
    free(bigram_table);
    free(vocab_hash);
    // 把全部的临时文件合并
    return merge_files(fidcounter + 1); // Merge the sorted temporary files
}

// 查找某个命令行参数
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

int main(int argc, char **argv) {
    int i;
    real rlimit, n = 1e5;
    vocab_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
    file_head = malloc(sizeof(char) * MAX_STRING_LENGTH);
    
    if (argc == 1) {
        printf("Tool to calculate word-word cooccurrence statistics\n");
        printf("Author: Jeffrey Pennington (jpennin@stanford.edu)\n\n");
        printf("Usage options:\n");
        printf("\t-verbose <int>\n");
        printf("\t\tSet verbosity: 0, 1, or 2 (default)\n");
        printf("\t-symmetric <int>\n");
        printf("\t\tIf <int> = 0, only use left context; if <int> = 1 (default), use left and right\n");
        printf("\t-window-size <int>\n");
        printf("\t\tNumber of context words to the left (and to the right, if symmetric = 1); default 15\n");
        printf("\t-vocab-file <file>\n");
        printf("\t\tFile containing vocabulary (truncated unigram counts, produced by 'vocab_count'); default vocab.txt\n");
        printf("\t-memory <float>\n");
        printf("\t\tSoft limit for memory consumption, in GB -- based on simple heuristic, so not extremely accurate; default 4.0\n");
        printf("\t-max-product <int>\n");
        printf("\t\tLimit the size of dense cooccurrence array by specifying the max product <int> of the frequency counts of the two cooccurring words.\n\t\tThis value overrides that which is automatically produced by '-memory'. Typically only needs adjustment for use with very large corpora.\n");
        printf("\t-overflow-length <int>\n");
        printf("\t\tLimit to length <int> the sparse overflow array, which buffers cooccurrence data that does not fit in the dense array, before writing to disk. \n\t\tThis value overrides that which is automatically produced by '-memory'. Typically only needs adjustment for use with very large corpora.\n");
        printf("\t-overflow-file <file>\n");
        printf("\t\tFilename, excluding extension, for temporary files; default overflow\n");

        printf("\nExample usage:\n");
        printf("./cooccur -verbose 2 -symmetric 0 -window-size 10 -vocab-file vocab.txt -memory 8.0 -overflow-file tempoverflow < corpus.txt > cooccurrences.bin\n\n");
        return 0;
    }

    // 控制是否输出debug信息，有0、1、2共三个级别
    if ((i = find_arg((char *)"-verbose", argc, argv)) > 0) verbose = atoi(argv[i + 1]);
    // 对称性，0表示只用当前词左侧窗口作为上下文，1表示使用当前词左右两侧对称的两个窗口作为上下文，默认为1
    if ((i = find_arg((char *)"-symmetric", argc, argv)) > 0) symmetric = atoi(argv[i + 1]);
    // 窗口大小，用于获取上下文
    if ((i = find_arg((char *)"-window-size", argc, argv)) > 0) window_size = atoi(argv[i + 1]);
    // 词表文件，默认为vocab.txt
    if ((i = find_arg((char *)"-vocab-file", argc, argv)) > 0) strcpy(vocab_file, argv[i + 1]);
    else strcpy(vocab_file, (char *)"vocab.txt");
    // TODO
    if ((i = find_arg((char *)"-overflow-file", argc, argv)) > 0) strcpy(file_head, argv[i + 1]);
    else strcpy(file_head, (char *)"overflow");
    // 内存消耗的软限制，单位GB，默认值4，简单的启发式限制所以并不是极端准确的
    if ((i = find_arg((char *)"-memory", argc, argv)) > 0) memory_limit = atof(argv[i + 1]);
    
    // TODO
    /* The memory_limit determines a limit on the number of elements in bigram_table and the overflow buffer */
    /* Estimate the maximum value that max_product can take so that this limit is still satisfied */
    // 1073741824 = 2^30, 即1GB
    // 所以rlimit是record limit，是memory_limit指定的内存使用数目的85%所能存储的CREC共现记录的数目
    // 剩下的15%就是留给overflow用
    // memory_limit是一个粗略的估计，因为哈希表什么的数据结构也是用内存的
    rlimit = 0.85 * (real)memory_limit * 1073741824/(sizeof(CREC));
    // n的初始值为1e5
    // 0.1544313298这个数，搜了一下，在这里提到了：
    // http://numbers.computation.free.fr/Constants/Gamma/gammaFormulas.html
    // TODO 没搞懂具体的数学原理，但总之是迭代计算出一个合适的n，满足nlogn + 0.15n ~= rlimit
    while (fabs(rlimit - n * (log(n) + 0.1544313298)) > 1e-3) n = rlimit / (log(n) + 0.1544313298);
    // 计算得到预估的max_product
    max_product = (long long) n;
    overflow_length = (long long) rlimit/6; // 0.85 + 1/6 ~= 1
    
    /* Override estimates by specifying limits explicitly on the command line */
    // TODO
    if ((i = find_arg((char *)"-max-product", argc, argv)) > 0) max_product = atoll(argv[i + 1]);
    // TODO
    if ((i = find_arg((char *)"-overflow-length", argc, argv)) > 0) overflow_length = atoll(argv[i + 1]);
    
    // 构造共现矩阵
    return get_cooccurrence();
}

