# -*- coding: utf-8 -*-
import os
import json
import sys
import yaml
import pickle
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS

# Add GEAD code path
sys.path.append('../code/')
import AE
import process_utils
from gead import GEADUsage
from process_utils import CicDataLoader
from openai import OpenAI
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# 配置文件路径
MODEL_SAVE_DIR = 'models'
RULE_SAVE_DIR = os.path.join(os.path.dirname(__file__), 'rules')

@app.route('/api/get_rules', methods=['POST'])
def get_rules():
    model_path = request.json.get('model_path')
    rule_json_path = os.path.join(RULE_SAVE_DIR, f'{secure_filename(os.path.basename(model_path))}.json')
    
    if not os.path.exists(rule_json_path):
        return jsonify({'error': '规则文件不存在'}), 404
        
    with open(rule_json_path, 'r') as f:
        rules = json.load(f)
    # Convert numpy types to native Python types
    def convert_types(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        # 处理嵌套结构
        elif isinstance(obj, (np.ndarray, np.int64, np.float64)):
            return obj.tolist()
        return obj
    rules = convert_types(rules)
    
    return jsonify({'rules': rules})
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RULE_SAVE_DIR, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/logo')
def get_logo():
    try:
        return send_file('../doc/gead_logo.png', mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/rules', methods=['GET'], endpoint='get_rules_list')
def get_rules():
    try:
        model_path = request.args.get('model_path')
        if not model_path or not os.path.exists(model_path):
            return jsonify({'error': '模型文件不存在'}), 400
            
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        # 加载GEAD配置
        with open('../demo/demo.yml', 'r') as f:
            config = yaml.safe_load(f)
            
        # 提取规则
        gead_usage = GEADUsage(model, [], verbose=True, debug=False, **config['rt_params'])
        rules = gead_usage.get_rules()
        
        return jsonify({
            'success': True,
            'rules': rules
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/optimize_rules', methods=['POST'])
def optimize_rules():
    try:
        data = request.get_json()
        rules = data.get('rules')
        optimization_type = data.get('type')  # 'manual' or 'ai'
        
        if optimization_type == 'manual':
            # 处理人工优化的规则
            return jsonify({
                'success': True,
                'message': '规则已更新'
            })
        elif optimization_type == 'ai':
            # TODO: 调用大模型API进行规则优化
            return jsonify({
                'success': True,
                'optimized_rules': rules,
                'suggestions': ['建议1', '建议2']
            })
        else:
            return jsonify({'error': '无效的优化类型'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload_model', methods=['POST'])
def upload_model():
    try:
        # 获取上传的模型文件
        if 'model_file' not in request.files:
            return jsonify({'error': '请上传模型文件'}), 400
            
        model_file = request.files['model_file']
        model_path = os.path.join(MODEL_SAVE_DIR, model_file.filename)
        model_file.save(model_path)
        
        # 验证模型文件
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            # 放宽模型验证条件，只要能成功加载就认为是有效的模型文件
            if not isinstance(model, object):
                return jsonify({'error': '无效的模型文件格式'}), 400
        except Exception as e:
            return jsonify({'error': f'模型文件加载失败: {str(e)}'}), 400
            
        return jsonify({
            'success': True,
            'model_path': model_path
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    try:
        # 获取上传的数据文件
        if 'data_file' not in request.files:
            return jsonify({'error': '请上传数据文件'}), 400
        
        data_file = request.files['data_file']
        data_path = os.path.join('uploads', data_file.filename)
        os.makedirs('uploads', exist_ok=True)
        data_file.save(data_path)


        data_path = '/data1/hdq/Projects/GEAD/release/GEAD/web/models/Tuesday-WorkingHours.pcap_ISCX.csv.model'
        
        # 加载和预处理数据
        dl = CicDataLoader(improved=False)
        X_all, y_all = dl.load_data(data_path)
        feature_names = dl.feature_name  # 获取特征名称
        dataset, normalizer = dl.data_split_norm(
            X_all, y_all,
            norm='train_ben',
            n_train_ben=0.5,
            n_train=0.5,
            n_vali=0.25,
            n_test=0.25
        )
        
        # 训练模型
        process_utils.set_random_seed()
        model = AE.train_valid(
            dataset['X_train_ben'],
            dataset['X_train_ben'].shape[1],
            dataset['X_vali'],
            dataset['y_vali'],
            epoches=20,
            lr=1e-4,
            verbose=False
        )
        
        # 评估模型
        y_test_pred, y_test_rmse = AE.test(model, dataset['X_test'])
        roc_auc, best_thres = AE.eval_roc(
            y_test_rmse,
            dataset['y_test'],
            thres_max=model.thres*1.5,
            plot=False
        )
        
        # 保存模型
        model_path = os.path.join(MODEL_SAVE_DIR, f'{data_file.filename}.model')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        return jsonify({
            'success': True,
            'model_path': model_path,
            'roc_auc': float(roc_auc),
            'threshold': float(best_thres)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/extract_rules', methods=['POST'])
def extract_rules():
    try:
        # 加载配置和模型
        model_path = request.form.get('model_path')
        data_file = request.files.get('data_file')
        if not model_path or not os.path.exists(model_path):
            return jsonify({'error': '模型文件不存在'}), 400
        if not data_file:
            return jsonify({'error': '请上传数据文件'}), 400
            
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        # 加载GEAD配置
        with open('../demo/demo.yml', 'r') as f:
            config = yaml.safe_load(f)

        # 保存并加载数据文件
        data_path = os.path.join('uploads', data_file.filename)
        os.makedirs('uploads', exist_ok=True)
        data_file.save(data_path)

        print('loading feature_names')
        dl = CicDataLoader(improved=False)
        X_all, y_all = dl.load_data(data_path)
        feature_names = dl.feature_name
        dataset, normalizer = dl.data_split_norm(
            X_all, y_all,
            norm='train_ben',
            n_train_ben=0.5,
            n_train=0.5,
            n_vali=0.25,
            n_test=0.25
        )
        print('succuessfully load feature_names')
        
        # 清理临时文件
        os.remove(data_path)
        
        # rt_params, lc_params, aug_params, aug_settings = configs['rt_params'], configs['lc_params'], configs['aug_params'], configs['aug_settings']
        # # 提取规则
        gead_usage = GEADUsage(model, feature_names, verbose=True, debug=False, **config['rt_params'])
        gead_usage.build_params(
            dataset['X_train_ben'],
            config['lc_params'],
            config['aug_params'],
            config['aug_settings']
        )
        
        # 生成规则树图形对象
        from tree_utils import TreePlotter
        tp = TreePlotter(gead_usage.merged_tree, feature_names=feature_names)
        graph = tp.plot_reg_tree(gead_usage.ad_thres, branch_condition=True, normalizer=normalizer)
        
        
        # 保存为临时文件并返回路径
        rule_path = os.path.join(RULE_SAVE_DIR, f'{secure_filename(os.path.basename(model_path))}')
        graph.render(rule_path, format='png', cleanup=True)
        
        # 保存规则到JSON文件
        rules = gead_usage.get_rule_paths() 
        rule_json_path = os.path.join(RULE_SAVE_DIR, f'{secure_filename(os.path.basename(model_path))}.json')
        with open(rule_json_path, 'w') as f:
            json.dump(rules, f, indent=2)
        
        return jsonify({
            'success': True,
            'rule_path': rule_path,
            'rule_json': rule_json_path
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/view_rules/<path:filename>')
def view_rules(filename):
    try:
        normalized_filename = os.path.normpath(filename)
        rule_path = os.path.join(RULE_SAVE_DIR, normalized_filename)
        if not rule_path.startswith(RULE_SAVE_DIR):
            raise ValueError("Invalid file path")
        return send_file(rule_path, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_raw_json')
def get_raw_json():
    try:
        model_path = request.args.get('model_path')
        if not model_path:
            return jsonify(success=False, error="Missing model_path parameter"), 400
        
        normalized_model_path = os.path.normpath(model_path)
        if not normalized_model_path.startswith(RULE_SAVE_DIR):
            return jsonify(success=False, error="Invalid model_path parameter"), 400
        
        rule_json_path = os.path.join(RULE_SAVE_DIR, f'{secure_filename(os.path.basename(normalized_model_path))}.json')
        if not os.path.exists(rule_json_path):
            return jsonify(success=False, error="Rule file not found"), 404
        
        with open(rule_json_path) as f:
            return json.load(f)
    except Exception as e:
        return jsonify(success=False, error=str(e))

@app.route('/api/update_rule', methods=['POST'])
def update_rule():
    data = request.json
    model_path = data['model_path']
    rule_index = int(data['rule_index'])
    new_condition = data['new_condition']
    
    try:
        model_path = os.path.normpath(model_path)
        if not model_path.startswith(RULE_SAVE_DIR):
            raise ValueError("Invalid model path")
        rules = load_rules(model_path)
        if 0 <= rule_index < len(rules):
            rules[rule_index]['conditions'] = new_condition
            save_rules(model_path, rules)
            return jsonify({'status': 'success'})
        return jsonify({'status': 'error', 'message': 'Invalid rule index'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'Invalid rule index'})

@app.route('/api/delete_rule', methods=['POST'])
def delete_rule():
    data = request.json
    model_path = data['model_path']
    rule_index = int(data['rule_index'])
    
    try:
        model_path = os.path.normpath(model_path)
        if not model_path.startswith(RULE_SAVE_DIR):
            raise ValueError("Invalid model path")
        rules = load_rules(model_path)
        if 0 <= rule_index < len(rules):
            del rules[rule_index]
            save_rules(model_path, rules)
            return jsonify({'status': 'success'})
        return jsonify({'status': 'error', 'message': 'Invalid rule index'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
    return jsonify({'status': 'error', 'message': 'Invalid rule index'})

def load_rules(model_path):
    rule_json_path = os.path.join(RULE_SAVE_DIR, f'{secure_filename(os.path.basename(model_path))}.json')
    if not os.path.exists(rule_json_path):
        return []
    with open(rule_json_path, 'r') as f:
        return json.load(f)


def save_rules(model_path, rules):
    model_path = os.path.normpath(model_path)
    if not model_path.startswith(RULE_SAVE_DIR):
        raise ValueError("Invalid model path")
    rule_json_path = os.path.join(RULE_SAVE_DIR, f'{secure_filename(os.path.basename(model_path))}.json')
    with open(rule_json_path, 'w') as f:
        json.dump(rules, f, indent=2)

@app.route('/api/llm_optimize', methods=['POST'])
def llm_optimize():
    try:
        data = request.get_json()
        model_path = data['model_path']
        user_prompt = data['user_prompt']
        current_rules = data['current_rules']
        # print('current_rules', current_rules,'user_prompt',user_prompt)

        # 加载配置文件
        with open('config.yml', 'r') as f:
            config = yaml.safe_load(f)
        
        client = OpenAI(
            api_key=config['moonshot']['api_key'],
            base_url="https://api.moonshot.cn/v1",
        )

        completion = client.chat.completions.create(
            model = "moonshot-v1-8k",
            messages = [
                {"role": "system", "content": "你是一个网络安全规则优化专家，请根据以下规则和用户指令提供优化建议："},
                {"role": "user", "content": f"当前规则：{current_rules}\n优化指令：{user_prompt}"}
            ],
            temperature = 0.3,
        )
        
        res = completion.choices[0].message.content

        print('RES', res)

        return jsonify({
                'success': True,
                'suggestion': res
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'优化请求失败：{str(e)}'
        })


def load_rules(model_path):
    normalized_model_path = os.path.normpath(model_path)
    if not normalized_model_path.startswith(RULE_SAVE_DIR):
        raise Exception("Invalid model_path parameter")
    
    rule_json_path = os.path.join(RULE_SAVE_DIR, f'{secure_filename(os.path.basename(normalized_model_path))}.json')
    if not os.path.exists(rule_json_path):
        return []
    with open(rule_json_path, 'r') as f:
        return json.load(f)


def save_rules(model_path, rules):
    rule_json_path = os.path.join(RULE_SAVE_DIR, f'{os.path.basename(model_path)}.json')
    with open(rule_json_path, 'w') as f:
        json.dump(rules, f, indent=2)

if __name__ == '__main__':
    import socket
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='GEAD Web Application')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    args = parser.parse_args()
    
    # Try the specified port, or find an available one
    port = args.port
    max_attempts = 10
    
    for attempt in range(max_attempts):
        try:
            # Test if port is available
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('0.0.0.0', port))
            sock.close()
            break
        except OSError:
            print(f"Port {port} is in use, trying {port+1}...")
            port += 1
            if attempt == max_attempts - 1:
                print(f"Could not find an available port after {max_attempts} attempts.")
                sys.exit(1)
    
    print(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)