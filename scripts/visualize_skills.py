import argparse
import numpy as np
import joblib
import tensorflow as tf
import os

from sac.policies.hierarchical_policy import FixedOptionPolicy
from sac.misc.sampler import rollouts
from sac.misc import utils
from sac.envs.unity_env import UnityEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to the snapshot file.')
    parser.add_argument('--max-path-length', '-l', type=int, default=100)
    parser.add_argument('--speedup', '-s', type=float, default=1)
    parser.add_argument('--deterministic', '-d', dest='deterministic',
                        action='store_true')
    parser.add_argument('--no-deterministic', '-nd', dest='deterministic',
                        action='store_false')
    parser.add_argument('--separate_videos', type=bool, default=False)
    parser.set_defaults(deterministic=True)

    # unity_env args
    parser.add_argument('--idx', type=int, default=0)
    parser.add_argument('--no_graphics', type=bool, default=False)

    args = parser.parse_args()
    filename = os.path.splitext(args.file)[0] + '.avi'
    best_filename = os.path.splitext(args.file)[0] + '_best.avi'
    worst_filename = os.path.splitext(args.file)[0] + '_worst.avi'

    path_list = []
    reward_list = []

    
    with tf.Session() as sess:
        data = joblib.load(args.file)
        policy = data['policy']
        env = data['env']
        num_skills = data['policy'].observation_space.flat_dim - data['env'].spec.observation_space.flat_dim

        with policy.deterministic(args.deterministic):
            for z in range(num_skills):
                fixed_z_policy = FixedOptionPolicy(policy, num_skills, z)
                new_paths = rollouts(env, fixed_z_policy,
                                    args.max_path_length, n_paths=1,
                                    render=True, render_mode='rgb_array')
                path_list.append(new_paths)
                total_returns = np.mean([path['rewards'].sum() for path in new_paths])
                reward_list.append(total_returns)

                if args.separate_videos:
                    base = os.path.splitext(args.file)[0]
                    end = '_skill_%02d.avi' % z
                    skill_filename = base + end
                    utils._save_video(new_paths, skill_filename)
        
                import csv
                file_path = args.file.split('/')
                file_path = file_path[-1].split('.')[0]
                file_path = './data/' + file_path
                if not os.path.exists(file_path):
                    os.mkdir(file_path)
                print(file_path)
                with open(file_path + '/path%02d.csv' % z, 'w', newline='') as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    spamwriter.writerow(['X', '-X', 'Y', '-Y', 'X_speed', 'Y_speed'])
                    for ob in path_list[-1][0]['observations']:
                        spamwriter.writerow(ob)

        if not args.separate_videos:
            paths = [path for paths in path_list for path in paths]
            utils._save_video(paths, filename)

        print('Best reward: %d' % np.max(reward_list))
        print('Worst reward: %d' % np.min(reward_list))
        # Record extra long videos for best and worst skills:
        best_z = np.argmax(reward_list)
        worst_z = np.argmin(reward_list)
        for (z, filename) in [(best_z, best_filename), (worst_z, worst_filename)]:
            fixed_z_policy = FixedOptionPolicy(policy, num_skills, z)
            new_paths = rollouts(env, fixed_z_policy,
                                 3 * args.max_path_length, n_paths=1,
                                 render=True, render_mode='rgb_array')
            utils._save_video(new_paths, filename)
        env.terminate()

if __name__ == "__main__":
    main()