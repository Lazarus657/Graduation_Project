CJSON代码

```c
Version代码
cJSON_Version()
define CJSON_PUBLIC(type)   __declspec(dllexport) type CJSON_STDCALL
   //宏定义
define CJSON_STDCALL __stdcall
  //__stdcall约定，参数按照从右到左的顺序入栈，被调用的函数在返回前清理传送参数的栈，函数参数个数固定
__declspec(dllimport)
#作用是什么省掉def文件中手工定义导出哪些函数的一个方法 用在win32 DLL上面不太懂
```

```c
sprintf()
#函数声明
int sprintf(char *str, const char *format, ...) 
#str是一个指向字符数组的指针，数组存了c字符
/字符串常量
 sprintf(version, "%i.%i.%i", CJSON_VERSION_MAJOR, CJSON_VERSION_MINOR, CJSON_VERSION_PATCH);
```

```
create_objects->struct cJSON
#这里结构体为什么要递归定义
#无论什么样的储存类型都要占用四个字节，这个递归定义的struct决定了链表的寻址操作
#Cjson实际上是一个链表
#结构体指针，这里涉及到结构体引用成员变量的方式 1.结构体变量名.成员名 2.使用指针
create_objects->struct record
#结构体数组使用 说白了就是定义多个结构体然后用数组形式储存
voliate修饰变量，变量可能会意想不到发生变化

create_objects->cJSON_CreateObject(void)->cJSON_New_Item()
```

结构体指针的使用

```c
# include <stdio.h>
# include <string.h>
struct AGE
{
    int year;
    int month;
    int day;
};
struct STUDENT
{
    char name[20];  //姓名
    int num;  //学号
    struct AGE birthday;  //生日
    float score;  //分数
};
int main(void)
{
    struct STUDENT student1; /*用struct STUDENT结构体类型定义结构体变量student1*/
    struct STUDENT *p = NULL;  /*定义一个指向struct STUDENT结构体类型的指针变量p*/
    p = &student1;  /*p指向结构体变量student1的首地址, 即第一个成员的地址*/
    strcpy((*p).name, "小明");  //(*p).name等价于student1.name
    (*p).birthday.year = 1989;
    (*p).birthday.month = 3;
    (*p).birthday.day = 29;
    (*p).num = 1207041;
    (*p).score = 100;
    printf("name : %s\n", (*p).name);  //(*p).name不能写成p
    printf("birthday : %d-%d-%d\n", (*p).birthday.year, (*p).birthday.month, (*p).birthday.day);
    printf("num : %d\n", (*p).num);
    printf("score : %.1f\n", (*p).score);
    return 0;
}
```

为了使用的直观和方便，用指针引用结构体成员变量的方式：

- (*指针变量名).成员名
- 指针变量名->成员名
- 结构体变量名.成员名

```c
const char*以及 char*的区别
含义
char* 表示一个指针变量，并且这个变量是可以被改变的。
const char*表示一个限定不会被改变的指针变量
模式不同
char*是常量指针，地址不可以改变，但是指针的值可变。
const char*是指向常量的常量指针，地址与值均不可变。
指针指向的内容不同
char*指针指向的内容是可以改变的，是不固定的。赋值后在数据传递的过程中允许改变。
const char*指针指向的内容是固定的，不可改变的。对传入的参数，不会对指针指向的内容进行修改
```

```c
define elif endif defined的使用

define CJSON_PUBLIC(type)   __declspec(dllexport) type CJSON_STDCALL

elif defined(CJSON_IMPORT_SYMBOLS)
注意事项：
    1.ifdef和ifndef仅仅是一个宏参数的定义，不能使用表达式
示例：test1 test2被定义执行printf1，否则执行printf2
    if defined test1 || test2
        printf1("run")
    else
        printf2("error")
    endif
这样说感觉defined返回值是一个bool值  
```

```c
CJSON_PUBLIC(cJSON_bool) cJSON_AddItemToObject(cJSON *object, const char *string, cJSON *item)
{
    return add_item_to_object(object, string, item, &global_hooks, false);
}

```

```
指针作为函数的参数传递
必要性—>实参和形参的传递是单反向的，只能实参向形参传递，形参不能向实参传递，这种叫做引出拷贝传递
解决方法->如果拷贝地址就可以直接对地址所指向的内存单元进行操作
指针传递的好处还有一个就是提高效率
使用值传递的两个条件
    数据很小，比如就一个 int 型变量。
    不需要改变它的值，只是使用它的值。
```

```
assignment to expression with array type
错误原因c语言数组在定义之后不允许再次复制，只支持读写，修改也只能逐个进行修改,使用strcpy或者使用memcpy
```

```c
指针常量和常量指针
指针常量：
int *const p //指针常量
usage：

int a,b;
int *const p=&a;
*p=9;/ 可以实现，地址里的内容可以修改
p=&b；/ 不可以实现，地址不可以修改
 
常量指针
usage：
    
int a,b;
const int *p=&a;
p=&b;/操作成功
*p=6;/操作失败
```

```
size_t 相当于无符号整形变量 在32位系统上是32位
```

```
allocate
deallocate
reallocate
这里涉及到c++内存管理
侯捷老师
```

```c
static cJSON *cJSON_New_Item(const internal_hooks * const hooks)
/这里声明了cjson指针函数 返回值是一个cjson结构体 传入值是一个内存钩子

```

[关于cJson结构的介绍](https://blog.csdn.net/weixin_46571142/article/details/108830462?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-5-108830462-blog-115431235.235^v26^pc_relevant_default_base1&amp;spm=1001.2101.3001.4242.4&amp;utm_relevant_index=8)

>
>
>typora插入超链接 Ctrl + K

```c
static unsigned char* cJSON_strdup(const unsigned char* string, const internal_hooks * const hooks)
 
strlen 和sizeof的使用
/这里为什么定义的时候要单独加这一条传入变量时（const unsigned char * string)
length = strlen((const char*)string) + sizeof("");
```

```c
unsigned char * 和 char *的关系 以及sizeof strlen关键字的使用
  
```

```c
static 关键字的使用
    1.普通局部变量的使用(不加static) 说明 在一个函数内部定义的变量 编译器一般不对普通的局部变量进行初始化，除非对他进行显式声明
    静态局部变量使用static修饰定义 说明 声明没有进行赋初值的时候编译器也会初始化为0 静态局部变量存储于全局的数据区 函数调用结束 它的值也不会变化 
    总结 静态局部变量的效果跟全局变量有一拼，但是位于函数体内部，就极有利于程序的模块化了
    2.全局变量 静态全局变量仅对当前文件可见，其他文件不可访问，其他文件可以定义与其同名的变量，两者互不影响
    3.函数 静态函数的使用说明
    静态函数只能在声明它的文件中可见，其他文件不能引用该函数
	不同的文件可以使用相同名字的静态函数，互不影响
4.面向对象
```

```c
利用char *和 char[]去定义一个字符串
    两种方式的区别就是char[]定义的字符串是可以被修改的 char*字符串内容是不可以修改的
```

