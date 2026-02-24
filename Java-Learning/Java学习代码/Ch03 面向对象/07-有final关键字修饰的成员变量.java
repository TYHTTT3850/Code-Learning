public class Dog {
    final String name = "AAA";
    final int age = 4;

    // 只能有无参构造方法，因为有参构造方法需要修改常量记录的内容
    public Dog() {
    }

    //public Dog(String name,int age) {
        //this.name = name; //报错，因为name是final修饰的常量，不能修改它记录的内容
        //this.age = age; //报错，因为age是final修饰的常量，不能修改它记录的内容
    //}

    //public void setName(String name) {
        //this.name = name;
    //} //报错，因为name是final修饰的常量，不能修改它记录的内容

    //public void setAge(int age) {
        //if (age < 0 || age > 15) {
            //System.out.println("年龄不合法");
        //}
        //else{
            //this.age = age;
        //}
    //} //报错，因为age是final修饰的常量，不能修改它记录的内容
}