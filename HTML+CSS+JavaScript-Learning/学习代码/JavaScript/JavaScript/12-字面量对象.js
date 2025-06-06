
    /*属性都是成对出现的，包括属性名和值，它们之间使用英文 : 分隔。多个属性之间使用英文 , 分隔
    属性就是依附在对象上的变量
    属性名可以使用 "" 或 ''，一般情况下省略，除非名称遇到特殊符号如空格、中横线等*/

    // 声明对象对象型变量，使用一对花括号
    // 例如通过对象描述一个人的数据信息
    let person={
        name:"1111",
        age:12,
        gender:"male"
    }

    // 可以使用 . 或 [] 获得对象中属性对应的值
    console.log(person.name)
    console.log(person.age)
    console.log(person["gender"])

    //可以为对象动态添加属性，动态添加与直接定义是一样的，只是语法上更灵活
    person.height=180
    person["weight"]=75

    // 声明对象，并添加了若干方法后，可以使用 . 或 [] 调用对象中函数
    let block={
        height:10,
        width:10,
        getArea:function(){return this.height*this.width} //使用 this 访问自身的属性
    }
    console.log(block.getArea())

    //同样可以动态为对象添加方法，动态添加与直接定义是一样的，只是语法上更灵活。

    //注：无论是属性或是方法，同一个对象中出现名称一样的，后面的会覆盖前面的。

    // 可以使用 for 循环遍历对象
    for (let key in person) {
        console.log(key, person[key]);
    }