package org.example.springbootlearning;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

/*
RestFul请求：
POST：新增
GET：查询
PUT：修改
DELETE：删除
*/

@SpringBootApplication
@RestController
@RequestMapping("/index")
public class Application {

    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }

//    @GetMapping
//    public String index() {
//        return "GET无参请求API";
//    }

    @GetMapping("{id}")//RestFul请求，参数通过路径的方式传递，路径中的参数用{}包裹起来
    public String index(@PathVariable String id) {//想要读前端传过来的参数，必须加上@PathVariable注解修饰方法中的形参
        return "GET RestFul请求API，参数id=" + id;
    }

    @GetMapping//普通请求，参数通过?key=value的方式传递，多个参数之间用&符号连接
    public String index2(@RequestParam String id, @RequestParam String name) {//想要读前端传过来的参数，必须加上@RequestParam注解修饰方法中的形参
        return "GET普通请求API，参数id=" + id + "，参数name=" + name;
    }//相同的注解，方法必须加上不同的请求路径，否则会报错，所以第一个无参请求要注释掉。

    @PostMapping
    public String save(@RequestBody Map<String, String> map) {
        System.out.println(map.toString());
        return "POST请求接收成功";
    }

    @PutMapping("{id}")
    public String update(@PathVariable Long id, @RequestBody Map<String, String> map) {
        System.out.println("id=" + id);
        System.out.println(map.toString());
        return "PUT请求接收成功";
    }

    @DeleteMapping("{id}")
    public String delete(@PathVariable Long id) {
        System.out.println("id=" + id);
        return "DELETE请求接收成功";
    }
}