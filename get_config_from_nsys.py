import sqlite3
import pandas as pd
import numpy as np
import sys
import os
import math
import argparse
import csv
import copy


def get_df_from_sqlite(sql_path):
    conn = sqlite3.connect(sql_path)
    cur = conn.cursor()

    try:
        cur.executescript("""
            ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD "duration[ns]" TEXT;
            ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD kernelName TEXT;
            
            UPDATE CUPTI_ACTIVITY_KIND_KERNEL
            SET "duration[ns]" = end - start;

            UPDATE CUPTI_ACTIVITY_KIND_KERNEL
            SET kernelName = (
                SELECT StringIds.value AS shortName
                FROM StringIds
                WHERE CUPTI_ACTIVITY_KIND_KERNEL.shortName = StringIds.id
            )
        """)
    except sqlite3.Error as err:
        if "duplicate" in ' '.join(err.args):
            pass
        else:
            print(' '.join(err.args))

    # Column names : ['nvtxRange', 'kernelName', 'duration[us]', 'wave', 
    #                 'gridX', 'gridY', 'gridZ', 'blockX', 'blockY', 'blockZ', 
    #                 'staticSharedMemory', 'dynamicSharedMemory', 'registersPerThread']

    # Number of dataframe rows : (number of launched cuda graphs) x (number of kernels in a cuda graph)
    df = pd.read_sql(f"""
        SELECT kernelName, "duration[ns]", gridX, gridY, gridZ
        FROM CUPTI_ACTIVITY_KIND_KERNEL as kernel_list
        ORDER BY kernel_list.start
    """, conn)
    df.reset_index(inplace=True)
    df['gridX'] = df['gridX'] * df['gridY'] * df["gridZ"]
    df.rename(columns={'kernelName':'Name', 'gridX':'SM_usage', 'duration':'Duration'}, inplace=True)
    df = df.drop(columns = ['index', 'gridY', 'gridZ'])
    dummy_data = [-1 for dummy in range(len(df))]
    df.insert(loc=1, column="Profile", value=dummy_data)
    df.insert(loc=2, column="Memory_footprint", value=dummy_data)
    df = df[['Name','Profile','Memory_footprint','SM_usage','Duration']]
    df = df.replace(['vectorized_elementwise_kernel'], "void at::native::vectorized_elementwise_kernel")
    df.to_csv(f"/workspace/orion{sql_path.split('.')[1]}", index=False)
    conn.close()

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    args = parser.parse_args()

    train_model_trace = get_df_from_sqlite(args.path)
    print(train_model_trace)