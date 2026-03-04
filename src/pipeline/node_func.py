from collections import defaultdict
import copy
import logging
import os
import random
import time
from typing import Any, Dict
from pathlib import Path
import json
from evaluate import major_voting
from pipeline.utils import node_decorator
from pipeline.pipeline_manager import PipelineManager
from database_manager import DatabaseManager
from arctic_manager import ArcticManager
from llm import model_chose
from prompt import *
from util import extract_sql_from_text, extract_rule_from_text, execute_sql, get_last_node_result, get_filter_schema_from_sqls
from util import extract_filtered_ddl, format_table_column_name, process_redundant_columns
import sqlglot


MAX_RETRIES= 3

@node_decorator(check_schema_status=False)
def schema_linking(task: Any, execution_history: Dict[str, Any]) -> Dict[str, Any]:
    config, node_name=PipelineManager().get_model_para()
    print(f"{node_name=}, {type(execution_history)=}, {type(task)=}")
    paths=DatabaseManager()
    chat_model = model_chose(node_name,config["engine"])  

    db_id = task.db_id
    question_id = task.question_id
    execute_history = task.execute_history
    sqlite_dir=paths.db_path

    print(sqlite_dir)

    pred_sqls = []
    execute_responses = []
    for k in range(config['n']):  # 一共生成n组结果，每组结果运行重试3次
        temperature = config["temperature"][k % len(config["temperature"])]
        content_input = get_filter_ddl_agent_prompt(task.db_desc, task.question)
        messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a data science expert. Below, you are provided with a database schema and a natural"
                        " language question. Your task is to understand the schema and generate a valid SQL query to"
                        " answer the question."
                    ),
                },
                {
                    "role": "user", 
                    "content": content_input
                }
            ]

        for att in range(MAX_RETRIES):   # TODO 这个错误控制应该不是这么写的
            try:
                llm_response = chat_model.get_ans(messages, temperature=temperature)      #添加温度参数
                print(f"现在温度是：{temperature}\n")
                # 提取SQL语句
                sqls = extract_sql_from_text(llm_response)

                # 生成错误的列需要纠正
                execute_response = execute_sql(sqls[-1].strip(), sqlite_dir, execute_history)
                if execute_response[0] == 'Execute Failed':
                    messages.append({
                        "role": "user", 
                        "content": f"The previous SQL execution failed with the following error:\n{execute_response[1]}\nPlease correct the SQL and try again."
                    })
                    raise Exception(str(execute_response))

                pred_sqls.append(sqls[-1].strip())
                execute_responses.append(execute_response)
                break

            except Exception as e:
                print(f"第{att + 1}次尝试失败，错误信息：{str(e)}")
                
                # 如果是最后一次尝试，抛出异常
                if att == MAX_RETRIES - 1:
                    print(f"经过{MAX_RETRIES}次尝试后仍然失败，返回空列表")
                    pred_sqls.append("")
                    execute_responses.append("")
                else:                
                    # 等待一段时间再重试（可选）
                    print(f"等待1秒后进行第{att + 2}次尝试...")
                    time.sleep(1)

    response = {
        "sqls": pred_sqls,
        "executions": execute_responses
    }
    return response


@node_decorator(check_schema_status=False)
def schema_linking_info(task: Any, execution_history: Dict[str, Any]) -> Dict[str, Any]:
    config, node_name=PipelineManager().get_model_para()
    print(f"{node_name=}, {type(execution_history)=}, {type(task)=}")
    paths=DatabaseManager()

    db_id = task.db_id
    execute_history = task.execute_history
    question_id = task.question_id
    sqlite_dir=paths.db_path
    chat_model = model_chose(node_name,config["engine"])  

    pred_sqls = []
    execute_responses = []
    for k in range(config['n']):  # 一共生成n组结果，每组结果运行重试3次
        temperature = config["temperature"][k % len(config["temperature"])]  # 支持n>len(temperature)时循环使用
        content_input = get_filter_ddl_agent_prompt(task.db_desc_info, task.question)  # 这里不同，加上了ours信息
        messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a data science expert. Below, you are provided with a database schema and a natural"
                        " language question. Your task is to understand the schema and generate a valid SQL query to"
                        " answer the question."
                    ),
                },
                {
                    "role": "user", 
                    "content": content_input
                }
            ]
        for att in range(MAX_RETRIES):   # TODO 这个错误控制应该不是这么写的
            try:
                llm_response = chat_model.get_ans(messages, temperature=temperature)
                print(f"现在温度是：{temperature}\n")
                # 提取SQL语句
                sqls = extract_sql_from_text(llm_response)

                # 生成错误的列需要纠正
                execute_response = execute_sql(sqls[-1].strip(), sqlite_dir, execute_history)
                if execute_response[0] == 'Execute Failed':
                    messages.append({
                        "role": "user", 
                        "content": f"The previous SQL execution failed with the following error:\n{execute_response[1]}\nPlease correct the SQL and try again."
                    })
                    raise Exception(str(execute_response))

                pred_sqls.append(sqls[-1].strip())
                execute_responses.append(execute_response)
                break

            except Exception as e:
                print(f"第{att + 1}次尝试失败，错误信息：{str(e)}")
                
                # 如果是最后一次尝试，抛出异常
                if att == MAX_RETRIES - 1:
                    print(f"经过{MAX_RETRIES}次尝试后仍然失败，返回空列表")
                    pred_sqls.append("")
                    execute_responses.append("")
                else:                
                    # 等待一段时间再重试（可选）
                    print(f"等待1秒后进行第{att + 2}次尝试...")
                    time.sleep(1)
    
    response = {
        "sqls": pred_sqls,
        "executions": execute_responses
    }
    return response


def sql_generation_tool(draft_sql, task, chat_model):
    paths=DatabaseManager()
    sqlite_dir=paths.db_path
    try:
        expression = sqlglot.parse_one(draft_sql, dialect='sqlite')
        columns = expression.find_all(sqlglot.exp.Column)

        columns_used = set(str(col) for col in columns)
        table_names_only = set()
        for table in expression.find_all(sqlglot.exp.Table):
            # 获取表的基本名称，不包含schema
            table_names_only.add(table.name)
        for alias in expression.find_all(sqlglot.exp.TableAlias):
            if alias.this and hasattr(alias.this, 'name'):
                table_names_only.add(alias.this.name)

        filter_column_list = []
        for t_column in columns_used:
            try:
                _, t_column = t_column.replace('"', '`').split('.')
            except:
                t_column = t_column.replace('"', "`")
            filter_column_list.append(t_column)
        # 检查一些字段命名是否合理
        filter_column_list = format_table_column_name(filter_column_list)
        table_names_only = format_table_column_name(table_names_only)
        print("过滤所有涉及的列：", filter_column_list)
        print("过滤所有涉及的表：", table_names_only)
        
        # 提取过滤后的DDL
        filtered_ddl = extract_filtered_ddl(task.db_desc, filter_column_list, table_names_only)

        #TODO 把相似列移到外面了。 处理冗余列，将匹配的列和表名添加到现有集合中
        columns_used, table_names_only = process_redundant_columns(columns_used, table_names_only, task.consistency_redundant_columns, task.inconsistency_redundant_columns)

        filter_column_list = []
        for t_column in columns_used:
            try:
                _, t_column = t_column.replace('"', '`').split('.')
            except:
                t_column = t_column.replace('"', "`")
            filter_column_list.append(t_column)
        # 检查一些字段命名是否合理
        filter_column_list = format_table_column_name(filter_column_list)
        table_names_only = format_table_column_name(table_names_only)
        print("相似列中，过滤所有涉及的列：", filter_column_list)
        print("相似列中，过滤所有涉及的表：", table_names_only)

        # 提取过滤后的DDL
        filtered_ddl = extract_filtered_ddl(task.db_desc_info, filter_column_list, table_names_only)
        print("过滤后的DDL:")
        print("=" * 50)
        print(filtered_ddl)
    except:
        filtered_ddl = task.db_desc_info


    # 第二次调用LLM，加knowledge确认
    content_input = get_generate_sql_agent_prompt(filtered_ddl, task.question, draft_sql, task.example)
    messages = [{
                    "role": "system",
                    "content": (
                        "You are a data science expert. Below, you are provided with a database schema and a natural"
                        " language question. Your task is to understand the schema and generate a valid SQL query to"
                        " answer the question."
                    ),
                },
                {
                    "role": "user", 
                    "content": content_input
                }

    ]
    llm_response = chat_model.get_ans_with_tool(messages, task.fd_list, sqlite_dir, task.execute_history, max_iterations=6)
    pred_sql = extract_sql_from_text(llm_response)[-1]
    rules = extract_rule_from_text(llm_response)[-1]

    return pred_sql, rules

    

@node_decorator(check_schema_status=False)
def sql_generation(task: Any, execution_history: Dict[str, Any]) -> Dict[str, Any]:
    print('question_id: ', task.question_id, ' enter sql_generation -----')

    config,node_name = PipelineManager().get_model_para()
    print(f"{node_name=}, {type(execution_history)=}, {type(task)=}")
    paths=DatabaseManager()
    chat_model = model_chose(node_name, config["engine"])

    schema_linking_execution = get_last_node_result(execution_history, "schema_linking")["executions"]
    schema_linking_sql = get_last_node_result(execution_history, "schema_linking")["sqls"]    
    schema_linking_with_info_execution = get_last_node_result(execution_history, "schema_linking_info")["executions"]
    schema_linking_with_info_sql = get_last_node_result(execution_history, "schema_linking_info")["sqls"]    
    
    all_execution = schema_linking_execution + schema_linking_with_info_execution
    all_sqls = schema_linking_sql + schema_linking_with_info_sql
    
    # 相同运行结果统一
    execution_res = []
    for execution in all_execution:
        if 'The execution' in execution[1]:
            execution_res.append(execution[1].split('The execution')[1])
        else:
            execution_res.append(execution[1])  # 当执行不成功时

    # # 用字典保存相同值的所有下标
    index_map = defaultdict(list)
    for i, v in enumerate(execution_res):
        index_map[v].append(i)
    
    print(f'一共有{len(index_map)}种执行结果: {index_map.values()}')

    print('%'*30)
    print(all_execution[0][0])
    if len(index_map) == 1 and all_execution[0][0] == 'Execute Success':
        pred_sqls = all_sqls  # 不用执行了，直接赋值，相当于终止流程
        rules = [""]*len(pred_sqls)
        print('跳过，直接赋值！')

    else:
        pred_sqls = []
        rules = []
        for sql in all_sqls:
            for att in range(MAX_RETRIES):   # TODO 这个错误控制应该不是这么写的
                try:
                    generation_sql, rule = sql_generation_tool(sql, task, chat_model)
                    pred_sqls.append(generation_sql)
                    rules.append(rule)
                    break
                except Exception as e:
                    print(f"第{att + 1}次尝试失败，错误信息：{str(e)}")
                    
                    # 如果是最后一次尝试，抛出异常
                    if att == MAX_RETRIES - 1:
                        print(f"经过{MAX_RETRIES}次尝试后仍然失败，返回空列表")
                        pred_sqls.append("")
                        rules.append("")
                    else:                
                        # 等待一段时间再重试（可选）
                        print(f"等待1秒后进行第{att + 2}次尝试...")
                        time.sleep(1) 

    print(pred_sqls)

    response = {
        "sqls": pred_sqls,
        "rules": rules
    }
    return response




# 用来函数对齐的
@node_decorator(check_schema_status=False)
def sql_style_refinement(task: Any, execution_history: Dict[str, Any]) -> Dict[str, Any]:
    print('question_id: ', task.question_id, 'enter sql_style_refinement -----')
    
    config,node_name = PipelineManager().get_model_para()
    print(f"{node_name=}, {type(execution_history)=}, {type(task)=}")
    paths = DatabaseManager()
    chat_model = model_chose(node_name, config["engine"])
    
    question_id = task.question_id
    execute_history = task.execute_history
    sqlite_dir = paths.db_path
    
    sql_generation_sqls = get_last_node_result(execution_history, "sql_generation")["sqls"]
    rules = get_last_node_result(execution_history, "sql_generation")["rules"]

    all_execution = []
    for sql in sql_generation_sqls:
        execute_response = execute_sql(sql, sqlite_dir, execute_history)
        all_execution.append(execute_response)
    execution_res = []
    for execution in all_execution:
        if 'The execution' in execution[1]:
            execution_res.append(execution[1].split('The execution')[1])
        else:
            execution_res.append(execution[1])  # 当执行不成功时
    
    # 用字典保存相同值的所有下标
    index_map = defaultdict(list)
    for i, v in enumerate(execution_res):
        index_map[v].append(i)
    
    print(f'一共有{len(index_map)}种执行结果: {index_map.values()}')

    pred_sqls = []
    for idx, (sql, rule) in enumerate(zip(sql_generation_sqls, rules)): 
        if 'dev' in sqlite_dir:
            content_input = get_style_sql_agent_dev_prompt(task.question, sql, rule)
        else:
            content_input = get_style_sql_agent_test_prompt(task.question, sql, rule)

        messages = [{
                        "role": "system",
                        "content": "You are a data science expert.",
                    },
                    {
                        "role": "user", 
                        "content": content_input
                    }
                ]
        for att in range(MAX_RETRIES):   
            try:
                llm_response = chat_model.get_ans(messages)
                generation_sql = extract_sql_from_text(llm_response)[-1]

                execute_response = execute_sql(generation_sql.strip(), sqlite_dir, execute_history)
                if execute_response[0] == 'Execute Failed':
                    messages.append({
                        "role": "user", 
                        "content": f"The previous SQL execution failed with the following error:\n{execute_response[1]}\nPlease correct the SQL and try again."
                    })
                    raise Exception(str(execute_response))                    
                pred_sqls.append(generation_sql)
                break
            except Exception as e:
                print(f"第{att + 1}次尝试失败，错误信息：{str(e)}")
                
                # 如果是最后一次尝试，抛出异常
                if att == MAX_RETRIES - 1:
                    print(f"经过{MAX_RETRIES}次尝试后仍然失败，返回空列表")
                    pred_sqls.append("")
                else:                
                    # 等待一段时间再重试（可选）
                    print(f"等待1秒后进行第{att + 2}次尝试...")
                    time.sleep(1)   

    response = {
        "sqls": pred_sqls
    }
    return response
    


@node_decorator(check_schema_status=False)
def sql_output_refinement(task: Any, execution_history: Dict[str, Any]) -> Dict[str, Any]:
    print('question_id: ', task.question_id, 'enter sql_output_refinement -----')

    config,node_name = PipelineManager().get_model_para()
    print(f"{node_name=}, {type(execution_history)=}, {type(task)=}")
    chat_model = model_chose(node_name, config["engine"])
    paths=DatabaseManager()

    question_id = task.question_id
    execute_history = task.execute_history
    sqlite_dir = paths.db_path

    style_sqls = get_last_node_result(execution_history, "sql_style_refinement")["sqls"]
    
    all_execution = []
    for sql in style_sqls:
        execute_response = execute_sql(sql, sqlite_dir, execute_history)
        all_execution.append(execute_response)
    execution_res = []
    for execution in all_execution:
        if 'The execution' in execution[1]:
            execution_res.append(execution[1].split('The execution')[1])
        else:
            execution_res.append(execution[1])  # 当执行不成功时
    
    # 用字典保存相同值的所有下标
    index_map = defaultdict(list)
    for i, v in enumerate(execution_res):
        index_map[v].append(i)
    
    print(f'一共有{len(index_map)}种执行结果: {index_map.values()}')


    pred_sqls = []
    for idx, sql in enumerate(style_sqls):
        content_input = get_output_sql_agent_prompt(task.question, sql)   # 先用大模型试试，不行再换成小模型
        messages = [{
                        "role": "system",
                        "content": "You are a data science expert.",
                    },
                    {
                        "role": "user", 
                        "content": content_input
                    }
                ]
        for att in range(MAX_RETRIES):
            try:
                llm_response = chat_model.get_ans(messages)
                pred_sql = extract_sql_from_text(llm_response)[-1]
                # 生成错误需要纠正
                execute_response = execute_sql(pred_sql.strip(), sqlite_dir, execute_history)
                if execute_response[0] == 'Execute Failed':
                    messages.append({
                        "role": "user", 
                        "content": f"The previous SQL execution failed with the following error:\n{execute_response[1]}\nPlease correct the SQL and try again."
                    })
                    raise Exception(str(execute_response))

                pred_sqls.append(pred_sql)
                break
            except Exception as e:
                print(f"第{att + 1}次尝试失败，错误信息：{str(e)}")
                
                # 如果是最后一次尝试，抛出异常
                if att == MAX_RETRIES - 1:
                    print(f"经过{MAX_RETRIES}次尝试后仍然失败，返回空列表")
                    pred_sqls.append("")
                else:                
                    # 等待一段时间再重试（可选）
                    print(f"等待1秒后进行第{att + 2}次尝试...")
                    time.sleep(1)

    response = {
        "sqls": pred_sqls
    }
    return response



@node_decorator(check_schema_status=False)
def sql_selection(task: Any, execution_history: Dict[str, Any]) -> Dict[str, Any]:
    print('question_id: ', task.question_id, 'enter sql_selection -----')

    config,node_name = PipelineManager().get_model_para()
    print(f"{node_name=}, {type(execution_history)=}, {type(task)=}")
    paths = DatabaseManager()
    
    sqlite_dir=paths.db_path

    style_refinement_sqls = get_last_node_result(execution_history, "sql_output_refinement")["sqls"]
    candidate_sqls = style_refinement_sqls
    sampling_num = len(candidate_sqls)
    db_files = [sqlite_dir] * sampling_num
    mj_pred_sqls = major_voting(db_files, candidate_sqls, sampling_num) # [[sql]]   

    response = {
        "candidate_sqls": candidate_sqls,
        "sqls": mj_pred_sqls[0] 
    }
    return response 
    
