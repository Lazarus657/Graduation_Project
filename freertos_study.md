C语言程序设计：现代方法



1.堆

```c
char heap_buf[1024]
int pose=0;
void *my_malloc(int size)
{
    int old_pose=pose;
    pose+=size;
    return &heap_buf[old_pose];
}
void *my_free(void *buf)

int main(void)
{
    char ch=65;// char ch='A'
    int i;
    char *buf=my_malloc(100);
    "怎样定义的buf"
    unsigned char uch=200;
    
    for (i=0;i<26;i++)
        buf[i]='A'+i;
    "断点观察"
}
"指针函数返回值是指针的函数，void指针函数，可以定义一个指针变量，但是不指定它是指向哪种类型的数据结构"
```

2.栈

```c
int a_fun(int val)
{
    int a=8;
    a+=val;
    b_fun();
    c_fun();
    return a;
}
void b_fun(void)
{
    
}

void c_fun(void)
{
    
}

int main(void)
{
    a_fun(46);
}
```

1.返回地址保存在哪里呢 保存在栈中

main->a_fun

1. LR寄存器=1地址
2. 调用a_fun

a_fun->b_fun

1. LR寄存器=2地址
2. 调用b _fun

函数将LR值保存在栈中

2.c函数开头

1. 划分栈 （LR寄存器，局部变量）
2. LR寄存器存入栈
3. 执行代码

3.BL main

1. LR=返回地址
2. 执行main

总结：就是进入函数时，下一条语句的地址交给LR寄存器 函数执行完成返回LR的地址，进行下一条

​	每个任务都要使用自己的栈，为什么main存着其他fun的地址



添加串口打印功能

- source insight clion