public class test {
    public static void main(String[] args) {
        /*
        变量调用：
            编译看左边，运行也看左边
        方法调用：
            编译看左边，运行看右边
        */
        Parent p = new Child();//父类引用指向子类对象
        /*
        调用变量：
            编译看左边：java编译成class文件时，看父类当中有没有这个变量，有则编译通过，没有则编译失败
            运行看左边：代码运行的时候，使用父类中的成员变量
        */
        System.out.println(p.name);
        //System.out.println(p.age);//父类没有age成员变量，报错

        /*调用方法：
            编译看左边：java编译成class文件时，看父类当中有没有这个方法，有则编译通过，没有则编译失败
            运行看右边：代码运行的时候，使用子类中的成员方法，如果没有重写父类方法，则使用父类中的成员方法
        */
        //p.childShow;//父类没有childShow方法，编译失败
        p.show1();//子类重写，运行时使用子类方法
        p.show2();//子类没有重写，运行时使用父类方法

        //所以多态有弊端：只能访问父类中有的方法和变量，不能访问子类中特有的方法和变量
        //解决方法：强制类型转换
        //但是强制类型转换时只能由爸爸转换成儿子，不能由爷爷转换为儿子。
        Child c = (Child)p;//父类引用强转成子类引用
        c.childShow();
        //转换前先判断
        Parent p2 = new Child();
        if(p2 instanceof Child) {
            Child c2 = (Child) p2;
            c2.childShow();
        }
    }

}