public class test{
    public static void main(String[] args){
        Person p1 = new Person("Alice", 30);//初始化一个对象
        Person p2 = new Person("Bob", 25);
        System.out.println(p1.getName() + " " + p1.getAge());//调用对象的成员方法
        System.out.println(p2.getName() + " " + p2.getAge());
    }
}

