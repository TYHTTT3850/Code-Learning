// 1、平方函数
const square = (x) => x * x; // 赋值给变量，使得变量成为一个函数对象
console.log(square(5)); // 输出：25

// 2️、加法函数（两个参数）
const add = (a, b) => a + b;
console.log(add(3, 4)); // 输出：7

// 3️、返回对象（对象必须用括号包起来）
const createUser = (name) => ({ name: name, isActive: true });
console.log(createUser('Alice')); 
// 输出：{ name: 'Alice', isActive: true }

// 4️、数组映射处理（每个值乘以 2）
const nums = [1, 2, 3, 4];
const doubled = nums.map((n) => n * 2); // 作为函数的参数传递
console.log(doubled); // 输出：[2, 4, 6, 8]

// 5️、无参数的箭头函数
const greet = () => console.log("Hello!");
greet(); // 输出：Hello!

// 6️、解构参数的函数
const showName = ({ name }) => console.log(name);
showName({ name: "Bob" }); // 输出：Bob
