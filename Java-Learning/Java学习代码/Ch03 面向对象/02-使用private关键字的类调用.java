public class test{
    public static void main(String[] args){
        Dog d1 = new Dog();
        d1.setName("小白");
        d1.setAge(3);
        System.out.println("名字：" + d1.getName() + " 年龄：" + d1.getAge());
        d1.eatBone();
    }
}

