import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

import sqlite3
from sqlite3 import Error
import pprint
import atexit
# from util import dec
from collections import defaultdict
from random import shuffle
import time
from queue import Queue


class SQLite:
    def __init__(self, db=None,check_same_thread=True, isolation_level='DEFERRED'):
        self.deferred_buffers = defaultdict(Queue)
        self.currently_processing = False
        self.conn = None
        if db is not None:
            self.connect(db,check_same_thread=check_same_thread,isolation_level=isolation_level)

    def connect(self,db, check_same_thread=True, isolation_level='DEFERRED'):
        self.db = db
        self.conn = sqlite3.connect('data/' + db + '.db', timeout=5, check_same_thread=check_same_thread,
                                    isolation_level=isolation_level)

    def disconnect(self):
        if self.conn is not None:
            # print("DISCONNECT FROM " + self.db)
            self.conn.close()
            self.conn = None

    def commit(self):
        self.conn.commit()

    def create_table(self,table_name,fields,drop=True):
        conn = self.conn
        c = conn.cursor()
        if drop:
            query = "DROP TABLE IF EXISTS "+table_name
            c.execute(query)
        query = "CREATE TABLE IF NOT EXISTS "+table_name+" ("+fields+")"
        c.execute(query)
        conn.commit()

    def create_index(self,index_name,table_name,fields, unique=False):
        conn = self.conn
        c = conn.cursor()
        query = "CREATE "
        if unique:
            query += "UNIQUE "
        query += "INDEX IF NOT EXISTS " + index_name + " on " + table_name + " (" + fields + ")"
        c.execute(query)
        conn.commit()

    def query(self,q, commit=True):
        c = self.conn.cursor()
        c.execute(q)
        modified = c.rowcount
        if commit:
            self.commit()
        return modified

    # def get_column_type(self, table, column):
    #     affinity_map = {
    #         'INTEGER':0,
    #         'INT':0,
    #         'NUMERIC':0,
    #         'REAL':0,
    #         'BLOB':2
    #     }
    #     if table not in self.type_mapping:
    #         c = self.conn.execute('PRAGMA table_info('+table+')')
    #         columns = c.fetchall()
    #         self.type_mapping[table] = {}
    #         for entry in columns:
    #             type = entry[2].upper()
    #             if type in affinity_map:
    #                 self.type_mapping[table][entry[1]] = affinity_map[type]
    #             else:
    #                 self.type_mapping[table][entry[1]] = 1 #text
    #     return self.type_mapping[table][column]

    def insert_kw(self, table, **kwargs):
        column_list = []
        value_list = []
        command_list = {'commit': False, 'connection': None, 'ignore':False, 'values':None}
        for key, value in kwargs.items():
            if key in command_list:
                command_list[key] = value
                continue
            column_list.append('`'+key+'`')
            if value is None:
                value_list.append(None)
            else:
                try:
                    check = float(value)
                    value_list.append(str(value))
                except Exception:
                    if type(value) == bytes:
                        value_list.append(sqlite3.Binary(value))
                    else:
                        value_list.append("'" + value + "'")

        error_mode = 'REPLACE'
        if command_list['ignore']:
            error_mode = 'IGNORE'

        conn_to_use = self.conn
        if command_list['connection'] is not None:
            conn_to_use = command_list['connection']

        if command_list['values'] is not None:
            value_list = []
            placeholder_list = []
            for value in command_list['values']:
                if type(value) == bytes:
                    value_list.append(sqlite3.Binary(value))
                else:
                    value_list.append(str(value))
                # try:
                #     value_list.append(str(value))
                # except Exception:
                #     value_list.append(sqlite3.Binary(value))
                placeholder_list.append("?")
            # query = "INSERT OR " + error_mode + " INTO " + table + " VALUES (" + ",".join(value_list) + ")"
            query = "INSERT OR " + error_mode + " INTO " + table + " VALUES (" + ",".join(placeholder_list) + ")"
            c = conn_to_use.execute(query,value_list)
        else:
            query = "INSERT OR "+error_mode+" INTO " + table + " (" + ",".join(column_list) + ") VALUES (" + ",".join(value_list) + ")"
            c = conn_to_use.execute(query)



        try:

            if command_list['commit']:
                conn_to_use.commit()
            return c.rowcount
        except Error as e:
            print("insert_kw error ", e, "table",table,"kwargs",kwargs)
            exit(0)



    def deferred_insert(self, table, values):
        buffer = self.deferred_buffers[table]
        if values is not None:
            converted_values = []
            for value in values:
                if type(value) == bytes:
                    converted_values.append(sqlite3.Binary(value))
                else:
                    converted_values.append(str(value))
            # buffer.append(converted_values)
            buffer.put(converted_values)
        # print('deferred',table,len(self.deferred_buffers[table]))


    def process_deferred_inserts(self,min_count, max_count_total=100, error_mode='IGNORE', single_table=False):
        # print("[",end='')
        if self.currently_processing:
            print("CURRENTLY IN INSERTS!!!")
        self.currently_processing = True
        tables = list(self.deferred_buffers.keys())
        len_list = []
        for table in tables:
            # len_list.append((table,len(self.deferred_buffers[table])))
            len_list.append((table,self.deferred_buffers[table].qsize()))
        len_list.sort(key=lambda tup: -tup[1])

        total_cnt = 0
        table_cnt = 0
        exec_time = 0
        self.conn.execute("BEGIN TRANSACTION")
        for table, _ in len_list:
            buffer = self.deferred_buffers[table]
            # if len(self.deferred_buffers[table]) >= min_count:
            if self.deferred_buffers[table].qsize() >= min_count:
                # print(table, "ACTUALLY INSERTING", len(self.deferred_buffers[table]))
                # placeholder_list = ["?"] * len(buffer[0])
                values = self.deferred_buffers[table].get()
                placeholder_list = ["?"] * len(values)
                # query = "INSERT OR " + error_mode + " INTO " + table + " VALUES (" + ",".join(placeholder_list) + ")"
                query = "INSERT INTO " + table + " VALUES (" + ",".join(placeholder_list) + ")"
                query_list = []

                query_list.append(values)
                total_cnt += 1

                # current_length = len(buffer) #buffer may grow while processing it! In that case this may never finish unless short-circuited.
                while not self.deferred_buffers[table].empty() and total_cnt < max_count_total:
                    values = self.deferred_buffers[table].get()
                    query_list.append(values)
                    total_cnt += 1
                # for values in buffer:
                #     query_list.append(values)
                #     blob_size_total += len(values[2])
                #     cnt += 1
                #     total_cnt += 1
                #     if total_cnt == max_count_total:
                #         break


                table_cnt += 1
                t = time.time()
                try:
                    self.conn.executemany(query, query_list)
                except:
                    raise NotImplementedError("Couldn't handle query "+query+", values"+str(query_list))
                exec_time += (time.time()-t)
                # print(table, "ACTUALLY INSERTED",cnt)
                # self.deferred_buffers[table] = self.deferred_buffers[table][cnt:]
            if total_cnt >= max_count_total:
                break
            if single_table:
                break

        remaining_cnt = 0
        tables = list(self.deferred_buffers.keys())
        for table in tables:
            # remaining_cnt += len(self.deferred_buffers[table])
            remaining_cnt += self.deferred_buffers[table].qsize()
        self.conn.execute("COMMIT")
        self.currently_processing = False
        # print("]",end='')
        return table_cnt, total_cnt, remaining_cnt, exec_time



        # error_mode = 'REPLACE'
        # if not replace:
        #     error_mode = 'IGNORE'
        # query = "INSERT OR " + error_mode + " INTO " + table + " VALUES (?)"
        # self.conn.executemany(query,value_list)




    def update_kw(self, table, where, **kwargs):
        # column_list = []
        # value_list = []
        pair_list = []
        command_list = {'commit': False, 'connection': None, 'ignore': False}
        for key, value in kwargs.items():
            if key in command_list:
                command_list[key] = value
                continue

            try:
                check = float(value)
                pair_list.append(key + " = " +str(value))
            except Exception:
                pair_list.append(key + " = " + "'" + value + "'")

        error_mode = 'REPLACE'
        if command_list['ignore']:
            error_mode = 'IGNORE'
        # if 'IGNORE' in command_list:
        #     error_mode = 'IGNORE'
        query = "UPDATE OR " + error_mode + " "+ table + " SET " + (",").join(pair_list) + " WHERE "+ where

        conn_to_use = self.conn
        if command_list['connection'] is not None:
            conn_to_use = command_list['connection']

        try:
            c = conn_to_use.cursor()
            c.execute(query)
            # if command_list['commit']:
            if command_list['commit']:
                conn_to_use.commit()
            return c.rowcount
        except Error as e:
            print("update_kw error ", e, "table", table, "kwargs", kwargs)
            exit(0)




    def select(self,query, dict=False):
        def printer(b):
            print('converting', b)
            if b[0] == 'b' and b[1] in["'","\'"]:
                return b
            return b.decode('UTF-8')


        conn = self.conn
        def dict_factory(cursor, row):
            d = {}
            for idx, col in enumerate(cursor.description):
                d[col[0]] = row[idx]
            return d

        try:
            if dict:
                old_f = conn.row_factory
                conn.row_factory = dict_factory
            # conn.text_factory = printer
            c = conn.cursor()
            c.execute(query)
            dictres = c.fetchall()
            if dict:
                conn.row_factory = old_f
            return dictres
        except Error as e:
            print("Error ", e)
            exit(0)


    def attach(self, other_db_file, other_db_name):
        c = self.conn.cursor()
        c.execute("ATTACH '" + other_db_file + "' AS " + other_db_name)
        self.conn.commit()

