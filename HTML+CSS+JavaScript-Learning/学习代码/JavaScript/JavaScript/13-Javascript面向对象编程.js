// 基类：User（用户）
class User {
    // 私有字段（只能在类内部访问）
    #password;
  
    // 静态属性（属于类，而不是实例）
    static userCount = 0;
  
    // 构造函数：用于创建新用户
    constructor(username, email, password) {
      this.username = username;   // 公共属性
      this.email = email;
      this.#password = password;  // 私有属性
      User.userCount++;           // 记录用户数量
    }
  
    // 实例方法：显示用户信息
    displayInfo() {
      console.log(`User: ${this.username}, Email: ${this.email}`);
    }
  
    // Getter：读取私有密码（模拟）
    get maskedPassword() {
      return '*'.repeat(this.#password.length);
    }
  
    // Setter：修改密码，带简单校验
    set newPassword(pwd) {
      if (pwd.length < 6) {
        console.log("Password too short!");
      } 
      else {
        this.#password = pwd;
        console.log("Password updated.");
      }
    }
  
    // 静态方法：显示用户总数
    static getUserCount() {
      return User.userCount;
    }
  }
  
  // 子类：Admin（管理员）
  class Admin extends User {
    constructor(username, email, password, level) {
      // 调用父类构造函数
      super(username, email, password);
      this.level = level;  // 管理员级别
    }
  
    // 重写父类方法
    displayInfo() {
      console.log(`Admin: ${this.username}, Level: ${this.level}`);
    }
  
    // 新增方法：删除用户（这里只是模拟）
    deleteUser(user) {
      console.log(`${this.username} deleted user ${user.username}`);
      User.userCount--;
    }
  }
  
  // ---------------------------
  // 使用示例
  
  const user1 = new User("alice", "alice@example.com", "123456");// 生成对象加上 new 关键字
  user1.displayInfo();
  console.log("Masked password:", user1.maskedPassword);
  user1.newPassword = "abc";   // 太短
  user1.newPassword = "newsecurepass";
  
  const admin1 = new Admin("bob_admin", "admin@example.com", "admin123", "super");
  admin1.displayInfo();
  admin1.deleteUser(user1);
  
  // 访问静态属性和方法
  console.log("Total users:", User.getUserCount());
  