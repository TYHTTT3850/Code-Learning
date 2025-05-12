//1、事件流（捕获与冒泡）
    const flowGrandParent = document.getElementById('flow-grandparent');
    const flowParent = document.getElementById('flow-parent');
    const flowChild = document.getElementById('flow-child');
    
    flowGrandParent.addEventListener('click', () => console.log('Flow GrandParent - 冒泡'), false);//false表示事件监听注册在冒泡阶段，事件会在冒泡阶段触发
    flowParent.addEventListener('click', () => console.log('Flow Parent - 冒泡'), false);
    flowChild.addEventListener('click', () => console.log('Flow Child - 冒泡'), false);
    flowGrandParent.addEventListener('click', () => console.log('Flow GrandParent - 捕获'), true);//true表示事件监听注册在冒泡阶段，事件会在捕获阶段触发
    flowParent.addEventListener('click', () => console.log('Flow Parent - 捕获'), true);
    flowChild.addEventListener('click', () => console.log('Flow Child - 捕获'), true);
    
    //2、阻止事件冒泡
    const stopGrandParent = document.getElementById('stop-grandparent');
    const stopParent = document.getElementById('stop-parent');
    const stopChild = document.getElementById('stop-child');

    stopGrandParent.addEventListener('click', () => {
      console.log('GrandParent clicked');
    });
    stopParent.addEventListener('click', (ev) => {
      console.log('Parent clicked');
      ev.stopPropagation();//阻止子节点的事件向上冒泡导致父节点事件触发
    });
    stopChild.addEventListener('click', (ev) => {
      console.log('Child clicked');
      ev.stopPropagation();
    });
    
    //3、事件委托
    const delegateTable = document.getElementById('delegate-table');//将事件委托给 button 标签的的父级 table 标签
    delegateTable.addEventListener('click', function (ev) {
      if (ev.target.tagName === 'BUTTON') {
        console.log('点击了按钮：', ev.target.innerText);
      }
    });//通过事件冒泡的方式将点击 button 标签后返回的事件对象传递给父级 table 标签
