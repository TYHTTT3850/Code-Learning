public class Student extends Person {
    String grade;

    //构造方法
    public Student() {
        System.out.println("子类Student的空参构造执行了");
    }

    public Student(String name, int age, String grade) {
        super(name, age); //继承的父类的属性调用父类的有参构造。
        // 细节：如果想要访问父类的带参构造方法，必须在子类的构造方法中显式调用父类的构造方法，且必须放在第一行
        this.grade = grade;//子类独有的单独赋值
        System.out.println("子类Student的有参构造执行了");
    }
}
