/*
static 关键字要点：
    1.静态变量：被当前所有对象共享。共享：当一个对象修改了静态变量的值，其他对象也会受到影响。
    2.调用方法：
        (1).类名调用(推荐)
        (2).对象名调用(不推荐)
*/
public class Student{
    String name;
    int age;
    static String teacher;//所有学生共享一个老师
    // 静态变量随着类的加载而加载，先于对象而存在
}
