public class test {
    public static void main(String[] args) {
        Student stu1 = new Student("张三", 20, "一年级");
        System.out.println(stu1.name+","+stu1.age+","+stu1.grade);

        Student stu2 = new Student();//细节：当子类的构造方法没有显式调用父类的构造方法时，默认会调用父类的空参构造方法



    }
}