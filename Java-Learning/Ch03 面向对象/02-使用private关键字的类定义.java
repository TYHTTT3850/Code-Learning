public class Dog {
    //为了解决面向对象中的数据安全问题，所有的成员变量都要设置为私有的(private关键字)
    private String name;
    private int age;

    //提供一个公共的(public)成员方法来访问私有的成员变量
    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getAge() {
        return age;
    }

    public void setAge(int age) {
        if (age < 0 || age > 15) {
            System.out.println("年龄不合法");
        }
        else{
            this.age = age;
        }
    }

    public void eatBone(){
        System.out.println(this.age+"岁的"+this.name+"吃骨头");
    }
}
