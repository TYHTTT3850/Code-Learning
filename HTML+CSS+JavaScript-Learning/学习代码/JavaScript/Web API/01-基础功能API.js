// 获取元素
const title = document.getElementById("title");
const button = document.querySelector("#btn");
const paragraphs = document.querySelectorAll(".desc");
const box = document.getElementById("box");

// 修改文本内容
title.textContent = "标题已被修改";

// 修改样式
title.style.color = "blue";
box.style.border = "2px solid black";

// 操作类名
title.classList.add("highlight");      // 添加类
title.classList.remove("highlight");   // 移除类
title.classList.toggle("highlight");   // 有就去掉，没有就加上

// 设置和移除属性
button.setAttribute("disabled", "");
setTimeout(() => {
  button.removeAttribute("disabled"); // 2秒后解除禁用
}, 2000);

// 添加元素
const newPara = document.createElement("p");
newPara.textContent = "我是新创建的段落";
document.body.appendChild(newPara);

// 删除元素
setTimeout(() => {
  newPara.remove();
}, 4000);

// 替换元素
const newH1 = document.createElement("h1");
newH1.textContent = "新标题元素";
document.body.replaceChild(newH1, title);

// 点击按钮时修改段落内容
button.addEventListener("click", () => {
  paragraphs.forEach((p, i) => {
    p.textContent = `第 ${i + 1} 个段落已被点击修改`;
  });
  box.style.backgroundColor = "skyblue";
  });
