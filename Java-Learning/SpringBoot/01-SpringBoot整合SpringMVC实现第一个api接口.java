package org.example.springbootlearning;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

//需要引入SpringMVC的依赖
/*
引入方式：pom.xml文件中添加以下依赖：
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-web</artifactId>
</dependency>
*/

@SpringBootApplication
@RestController//这个注解表示这个类是一个控制器，可以处理HTTP请求。适合前后端分离的项目。
@RequestMapping("/index")//给这个类的所有方法添加一个公共的路径前缀，访问时需要加上这个前缀(例如http://localhost:8080/index/hello)
public class Application {

    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }

    //要接收请求，就需要创建方法
    @RequestMapping("/hello")//这个注解表示当访问/hello路径时，调用这个方法
    public String hello(){
        return "Hello, Spring Boot!";
    }
    //类上不注解@RequestMapping("/index")时，浏览器输入http://localhost:8080/hello时，页面出现Hello, Spring Boot!
    //类上注解了@RequestMapping("/index")时，浏览器输入http://localhost:8080/index/hello时，页面出现Hello, Spring Boot!

}
