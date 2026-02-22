public class test{
    public static void main(String[] args){
        Student s1 = new Student();
        s1.name = "张三";
        s1.age = 20;
        Student.teacher = "李老师";
        //s1.teacher = "李老师";//不推荐

        Student s2 = new Student();
        s2.name = "李四";
        s2.age = 22;

        System.out.println(s1.name + "的老师是：" + Student.teacher);
        System.out.println(s2.name + "的老师是：" + s2.teacher);

        //其中一个对象修改了静态变量teacher的值
        s2.teacher = "王老师";
        System.out.println(s1.name + "的老师是：" + s1.teacher);
        System.out.println(s2.name + "的老师是：" + s2.teacher);

    }
}

