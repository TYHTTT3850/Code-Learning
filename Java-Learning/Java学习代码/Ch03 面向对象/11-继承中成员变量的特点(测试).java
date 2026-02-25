public class test {
    public static void main(String[] args) {
        Child c = new Child();
        c.printValues();

        Parent p = c;
        System.out.println(p.x);//父类仍然访问父类的x
        System.out.println(c.x);//子类访问子类的x
    }
}