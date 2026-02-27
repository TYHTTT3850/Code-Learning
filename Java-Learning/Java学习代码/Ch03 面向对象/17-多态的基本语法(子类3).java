public class Admin extends Person{
    public Admin() {}

    public Admin(String name, String username, String password) {
        super(name, username, password);
    }

    @Override
    public void work() {
        System.out.println("管理员的工作是管理网站");
    }
}
