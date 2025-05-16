from bottle import Bottle, request, static_file, template, redirect
# 创建一个 Bottle 应用实例
app = Bottle()

# 路由：访问根路径 '/' 时触发的函数
@app.route('/')
def index():
    return "Hello, Bottle!"

# 路由：带参数的 URL，<name> 是动态部分
@app.route('/hello/<name>')
def greet(name):
    return f"Hello, {name}!"

# 路由：处理 GET 查询参数
@app.route('/search')
def search():
    keyword = request.query.keyword
    return f"Searching for: {keyword}"

# 路由：提供静态文件下载服务
@app.route('/static/<filename>')
def serve_static(filename):
    return static_file(filename, root='./static')

# 路由：使用模板渲染 HTML 页面
@app.route('/welcome/<name>')
def welcome(name):
    return template('Hello {{name}}, welcome!', name=name)

# 路由：重定向
@app.route('/redirectme')
def redirect_example():
    redirect('/') # 重定向到首页
