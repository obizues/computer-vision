import pandas as pd

df = pd.read_csv('data/eda_outputs/features_top_view.csv')
print('Columns:', df.columns.tolist())
print('Shape:', df.shape)
print('nose_dist in columns:', 'nose_dist' in df.columns)
if 'nose_dist' in df.columns:
    print('nose_dist quantile(0.25):', df['nose_dist'].quantile(0.25))
    print('Sample:')
    print(df[['frame_idx', 'nose_dist']].head(10))
    
    # Check a specific frame like 1827
    frame_1827 = df[df['frame_idx'] == 1827]
    if not frame_1827.empty:
        print(f'\nFrame 1827 nose_dist: {frame_1827["nose_dist"].values[0]}')
