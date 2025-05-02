function onClickHandler(e) {//事件处理程序
    console.log('点击位置：', e.clientX, e.clientY)
    }
    
const btn1 = document.getElementById('btn1')
const btn2 = document.getElementById('btn2')
    
// 将“click”事件监听器挂载到 btn1 和 btn2 上
btn1.addEventListener('click', onClickHandler);
btn2.addEventListener('click', function (e) {alert('按钮被点击！')} //按钮被按下时执行此函数
    )
    
// 取消事件监听
// btn1.removeEventListener('click', onClickHandler);