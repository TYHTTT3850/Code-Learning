public class test{
    public static void main(String[] args){
        OrderStatus o1 = OrderStatus.CREATED;
        System.out.println(o1.getCode());
        System.out.println(o1.getDescription());

        switch(o1){
            case CREATED -> System.out.println("订单已创建");
            case PAID -> System.out.println("订单已支付");
            case SHIPPED -> System.out.println("订单已发货");
            case COMPLETED -> System.out.println("订单已完成");
            case CANCELED -> System.out.println("订单已取消");
        }
    }
}

