import pandas as pd

df = pd.read_csv('data/eda_outputs/features_top_view.csv')
valid_all = df[(df['nose_dist'].notna()) & (df['nose_dist'] > 0.0)]
threshold = valid_all['nose_dist'].quantile(0.25)

print(f"25th percentile of nose_dist (valid frames only): {threshold:.2f}")
print()

for frame in [305, 978]:
    subset = df[df['frame_idx'] == frame][['frame_idx', 'nose_dist', 'b_nose_x', 'b_nose_y', 'w_nose_x', 'w_nose_y', 'is_close_interaction']]
    if not subset.empty:
        print(f'Frame {frame}:')
        print(subset.to_string())
        nose_dist = subset['nose_dist'].values[0]
        print(f"  Passes distance filter (< {threshold:.2f})? {nose_dist <= threshold if pd.notna(nose_dist) and nose_dist > 0 else 'INVALID'}")
    print()

# Check predictions too
pred_df = pd.read_csv('data/eda_outputs/batch_predictions.csv')
for frame in [305, 978]:
    pred = pred_df[pred_df['frame_idx'] == frame][['frame_idx', 'y_proba_close', 'y_pred']]
    if not pred.empty:
        print(f'Frame {frame} prediction:')
        print(pred.to_string())
    print()
