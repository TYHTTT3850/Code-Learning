/*
枚举是一个特殊的 JavaBean 类，这个类的对象为有限个
*/
public enum OrderStatus {
    // 这个类所有的对象
    //所有的枚举项默认都是 public static final 的，不需要自己加修饰符
    CREATED(0, "已创建"),
    PAID(1, "已支付"),
    SHIPPED(2, "已发货"),
    COMPLETED(3, "已完成"),
    CANCELED(4, "已取消");

    private final int code;
    private final String description;

    private OrderStatus(int code, String description) {//不让外部创建对象，所以用private
        this.code = code;
        this.description = description;
    }

    public int getCode() {
        return code;
    }

    public String getDescription() {
        return description;
    }
}
