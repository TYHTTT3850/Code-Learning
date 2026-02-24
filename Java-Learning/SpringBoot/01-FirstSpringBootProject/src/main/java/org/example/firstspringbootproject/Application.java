package org.example.firstspringbootproject;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication //这是一个注解(不是Python中的装饰器)，这个注解告诉框架：以这个类作为应用上下文的启动配置根节点。注解是 Java 提供的一种类型安全的元数据声明机制。
public class Application {

    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }

}
