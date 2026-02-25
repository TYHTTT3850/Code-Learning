public class test{
    public static void main(String[] args){
        Student s = new Student();
        s.name = "张三";
        s.age = 20;
        s.grade = 3;
        System.out.println(s.name + " " + s.age + " " + s.grade);
        s.eat();
        s.study();

        Teacher t = new Teacher();
        t.name = "李四";
        t.age = 40;
        t.subject = "数学";
        System.out.println(t.name + " " + t.age + " " + t.subject);
        t.eat();
        t.teach();
    }
}

