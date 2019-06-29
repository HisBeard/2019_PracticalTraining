### 人脸属性分析

#### 目录结构

    |-- 2019_PracticalTraining
        |-- controller/
            |-- __init__.py
            |-- login.py
        |-- fr/
            |-- __init__.py
            |-- admin.py
            |-- apps.py
        |-- pismap/
        |-- static/
            |-- css/
            |-- fonts/
            |-- js/
        |-- templates/
            |-- login.html

controller/：后台控制代码放这里面、处理网页的请求

fr/：人脸是别的库放这里面

\_\_init\_\_.py: 一个空文件，告诉 Python 该目录是一个 Python 包

settings.py: 该 Django 项目的设置/配置

urls.py: 该 Django 项目的 URL 声明; 一份由 Django 驱动的网站"目录"

static/：网页资源css、fonts、js放这里面

templates/：模板放这里面

manage.py: 一个实用的命令行工具，可让以各种方式与该 Django 项目进行交互

#### 开发环境

Python 3.6，Django，pytorch

#### 运行环境

1. 安装python 3.6，Django，pytorch

2. 在项目目录下输入python manage.py runserver localhost:8000 启动

3. localhost:8000/login 访问