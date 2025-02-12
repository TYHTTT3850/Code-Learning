## C++常见关键字

| 0      |     1      |      2       |        3         |      4      |    5     |
| ------ | :--------: | :----------: | :--------------: | :---------: | :------: |
| **1**  |    asm     |      do      |        if        |   return    | typedef  |
| **2**  |    auto    |    double    |      inline      |    short    |  typeid  |
| **3**  |    bool    | dynamic_cast |       int        |   signed    | typename |
| **4**  |   break    |     else     |       long       |   sizeof    |  union   |
| **5**  |    case    |     enum     |     mutable      |   static    | unsigned |
| **6**  |   catch    |   explicit   |    namespace     | static_cast |  using   |
| **7**  |    char    |    export    |       new        |   struct    | virtual  |
| **8**  |   class    |    extern    |     operator     |   switch    |   void   |
| **9**  |   const    |    false     |     private      |  template   | volatile |
| **10** | const_cast |    float     |    protected     |    this     | wchar_t  |
| **11** |  continue  |     for      |      public      |    throw    |  while   |
| **12** |  default   |    friend    |     register     |    true     |          |
| **13** |   delete   |     goto     | reinterpret_cast |     try     |          |

## ASCII码表

| **ASCII**值 | **控制字符** | **ASCII**值 | **字符** | **ASCII**值 | **字符** | **ASCII**值 | **字符** |
| ----------- | ------------ | ----------- | -------- | ----------- | -------- | ----------- | -------- |
| 0           | NUT          | 32          | (space)  | 64          | @        | 96          | 、       |
| 1           | SOH          | 33          | !        | 65          | A        | 97          | a        |
| 2           | STX          | 34          | "        | 66          | B        | 98          | b        |
| 3           | ETX          | 35          | #        | 67          | C        | 99          | c        |
| 4           | EOT          | 36          | $        | 68          | D        | 100         | d        |
| 5           | ENQ          | 37          | %        | 69          | E        | 101         | e        |
| 6           | ACK          | 38          | &        | 70          | F        | 102         | f        |
| 7           | BEL          | 39          | ,        | 71          | G        | 103         | g        |
| 8           | BS           | 40          | (        | 72          | H        | 104         | h        |
| 9           | HT           | 41          | )        | 73          | I        | 105         | i        |
| 10          | LF           | 42          | *        | 74          | J        | 106         | j        |
| 11          | VT           | 43          | +        | 75          | K        | 107         | k        |
| 12          | FF           | 44          | ,        | 76          | L        | 108         | l        |
| 13          | CR           | 45          | -        | 77          | M        | 109         | m        |
| 14          | SO           | 46          | .        | 78          | N        | 110         | n        |
| 15          | SI           | 47          | /        | 79          | O        | 111         | o        |
| 16          | DLE          | 48          | 0        | 80          | P        | 112         | p        |
| 17          | DCI          | 49          | 1        | 81          | Q        | 113         | q        |
| 18          | DC2          | 50          | 2        | 82          | R        | 114         | r        |
| 19          | DC3          | 51          | 3        | 83          | S        | 115         | s        |
| 20          | DC4          | 52          | 4        | 84          | T        | 116         | t        |
| 21          | NAK          | 53          | 5        | 85          | U        | 117         | u        |
| 22          | SYN          | 54          | 6        | 86          | V        | 118         | v        |
| 23          | TB           | 55          | 7        | 87          | W        | 119         | w        |
| 24          | CAN          | 56          | 8        | 88          | X        | 120         | x        |
| 25          | EM           | 57          | 9        | 89          | Y        | 121         | y        |
| 26          | SUB          | 58          | :        | 90          | Z        | 122         | z        |
| 27          | ESC          | 59          | ;        | 91          | [        | 123         | {        |
| 28          | FS           | 60          | <        | 92          | /        | 124         | \|       |
| 29          | GS           | 61          | =        | 93          | ]        | 125         | }        |
| 30          | RS           | 62          | >        | 94          | ^        | 126         | `        |
| 31          | US           | 63          | ?        | 95          | _        | 127         | DEL      |

## C++通过源码安装第三方库(MinGW)

使用的构建系统为：CMake。使用的编译器为：MinGW64。

将第三方库安装到统一的文件夹中以方便管理，这里以 `D:\Cpp_ThirdPartyLib` 为例，下载源码，假设其在文件夹 `D:\example-library` 下。

**1、在 `D:\example-library` 中创建一个专门用于构建的目录，比如 `build`**

```cmd

mkdir D:/example-library/build

```

**2、进入 `build` 文件夹**

```cmd

cd D:/example-library/build

```

**3、使用 CMake 配置项目**

使用 `cmake` 命令指定源代码目录 `D:/example-library` 和安装路径 `D:/Cpp_ThirdPartyLib`

```cmd

cmake .. -DCMAKE_INSTALL_PREFIX=D:/Cpp_ThirdPartyLib

```

`..` ：表示上级目录，即 `D:/example-library(源代码目录)`。
`-DCMAKE_INSTALL_PREFIX=D:/CppThirdPartyLib` ：指定库的安装路径为 `D:/Cpp_ThirdPartyLib` 。

**4、构建和安装库**

在构建目录中执行以下命令来编译并安装库：

```cmd

cmake --build . --config Release

```

```cmd

cmake --install .

```

这些命令将会：

编译库并将其安装到 `D:/CppThirdPartyLib` 目录下。
`cmake --install .` 会把库的文件(如头文件和库文件)复制到指定的安装目录。
