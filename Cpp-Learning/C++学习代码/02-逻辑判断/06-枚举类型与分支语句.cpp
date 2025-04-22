#include <iostream>
using namespace std;

enum class OrderStatus {
    Pending = 1,
    Processing = 2,
    Shipped = 3,
    Delivered = 4,
    Cancelled = 5
};

int main() {
    cout << "请输入订单状态编号：" << endl;
    cout << "1. 待处理\n2. 处理中\n3. 已发货\n4. 已送达\n5. 已取消" << endl;

    int input;
    cin >> input;

    OrderStatus status;

    switch (input) {
        case 1: status = OrderStatus::Pending; break;
        case 2: status = OrderStatus::Processing; break;
        case 3: status = OrderStatus::Shipped; break;
        case 4: status = OrderStatus::Delivered; break;
        case 5: status = OrderStatus::Cancelled; break;
        default:
            cout << "无效的输入。" << endl;
            return 1;
    }

    // 根据枚举状态输出提示
    switch (status) {
        case OrderStatus::Pending:
            cout << "订单待处理，请尽快审核。" << endl;
            break;
        case OrderStatus::Processing:
            cout << "订单正在处理，请耐心等待。" << endl;
            break;
        case OrderStatus::Shipped:
            cout << "订单已发货，物流信息已更新。" << endl;
            break;
        case OrderStatus::Delivered:
            cout << "订单已送达，感谢您的购买！" << endl;
            break;
        case OrderStatus::Cancelled:
            cout << "订单已取消。" << endl;
            break;
    }

    return 0;
}
