public class test{
    public static void main(String[] args){
        /*
        final 关键字修饰的变量叫做常量，常量在定义的时候必须初始化，并且不能修改。
        习惯上常量的名字全部大写，单词之间用下划线分隔。
        细节：
            基本数据类型：
                byte short int long float double char boolean
                变量记录的是真实的数据
            引用数据类型：
                除了上面四类八种，其他的都是引用数据类型。
                变量记录的是地址值。
            综上所述：
                final 修饰哪个变量，这个变量记录的内容不能修改。

        */

        // 定义一个基本数据类型的常量
        final int NUM = 10;
        // NUM = 20; //错误，不能修改常量记录的内容
        System.out.println(NUM);

        // 定义一个引用数据类型的常量
        final Student STU = new Student("张三", 20);
        // STU = new Student(); //错误，不能修改常量记录的内容
        STU.setName("李四"); //正确，因为并没有修改STU记录的内容(地址)
        System.out.println(STU.getName());

        // 如果不想要修改对象的属性，可以将属性也定义为final
        final Dog D = new Dog();
        // D.name = "BBB"; //错误，因为类定义中使用 final 修饰了 name 属性
        System.out.println(D.name);
    }
}

