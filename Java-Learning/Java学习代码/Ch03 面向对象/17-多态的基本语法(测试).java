/*
什么是多态：
    事物的多种形态
多态的表现形式：
    Parent parent = new Child();即父类引用指向子类对象
多态的前提：
    1.必须存在继承关系
    2.父类引用指向子类对象
    3.子类重写父类的方法(不重写的话代码不会报错，但享受不到好处)
多态的好处：
    1.方法中使用父类作为参数，可以接收父类对象和所有子类对象
    2.如果进行方法重写，利用多态调用方法，可以调用不同子类重写的方法，执行不同的操作
*/

public class test {
    /*
    学生类：
        属性：姓名，账号，密码
        行为：学生的工作是学习
    老师类：
        属性：姓名，账号，密码
        行为：老师的工作是教学
    管理员：
        属性：姓名，账号，密码
        行为：管理员的工作是管理网站
    */
    public static void register(Person person) {
        System.out.println("姓名为"+person.getName()+"用户名为"+person.getUsername()+"的用户注册成功"+"密码为"+person.getPassword());
        person.work();
    }
    public static void main(String[] args) {
        Student student = new Student("张三", "zhangsan", "123456");
        Teacher teacher = new Teacher("李四", "lisi", "654321");
        Admin admin = new Admin("王五", "wangwu", "111111");

        register(student);
        register(teacher);
        register(admin);
    }

}