using System;
using UnityEngine;

namespace Rigging.Solvers
{
    [Serializable]
    public class RBFMatrix
    {
        private int _rowCount;
        private int _columnCount;
        private float[,] _matrix;
        
        public RBFMatrix(int rows, int cols)
        {
            _rowCount = rows;
            _columnCount = cols;
            _matrix = new float[rows, cols];
        }

        public int GetRowSize()
        {
            return _rowCount;
        }

        public int GetColumnSize()
        {
            return _columnCount;
        }
        
        public float[] GetRowVector(int rowIndex)
        {
            float[] vec = new float[_columnCount];
            for (int i = 0; i < _columnCount; i++)
            {
                vec[i] = _matrix[rowIndex, i];
            }
            return vec;
        }
        
        public float[] GetColumnVector(int columnIndex)
        {
            float[] vec = new float[_rowCount];
            for (int i = 0; i < _rowCount; i++)
            {
                vec[i] = _matrix[i, columnIndex];
            }
            return vec;
        }

        public float GetElem(int row, int column)
        {
            return _matrix[row, column];
        }
        
        public void SetElem(int row, int column, float value)
        {
            _matrix[row, column] = value;
        }
        
        public RBFMatrix Transpose()
        {
            RBFMatrix result = new RBFMatrix(_rowCount, _columnCount);

            for (int i = 0; i < _rowCount; i++)
            {
                for (int j = 0; j < _columnCount; j++)
                {
                    result.SetElem(i, j, _matrix[j, i]);
                }
            }
            
            return result;
        }
        
        // Use Gaussian Elimination to solve the matrix, and set the values to w.
        public bool Solve(float[] y, ref float[] w)
        {
            if (_rowCount != _columnCount)
            {
                Debug.LogError("RBFMatrix is not square, can not solve.");
                return false;
            }

            for (int i = 0; i < _rowCount; i++)
            {
                // Find the row with the largest absolute value in the first
                // column and store the pivot index.
                float maxVal = _matrix[i, i];
                int pivot = i;
                bool swap = false;

                for (int j = i + 1; j < _rowCount; j++)
                {
                    if (Mathf.Abs(maxVal) < Mathf.Abs(_matrix[j, i]))
                    {
                        maxVal = _matrix[j, i];
                        pivot = j;
                        swap = true;
                    }
                }
                
                // Perform the row interchange if necessary.
                if (swap)
                {
                    for (int j = 0; j < _rowCount; j++)
                    {
                        w[j] = _matrix[pivot, j];
                        _matrix[pivot, j] = _matrix[i, j];
                        _matrix[i, j] = w[j];
                    }

                    // Swap the order of the values.
                    (y[pivot], y[i]) = (y[i], y[pivot]);
                }

                // Check if the matrix is singular.
                if (Mathf.Abs(_matrix[i, i]) < 0.0001f)
                {
                    return false;
                }
                
                // Perform the forward elimination.
                for (int j = i + 1; j < _rowCount; j++)
                {
                    float mult = _matrix[j, i] / _matrix[i, i];
                    for (int k = 0; k < _rowCount; k++)
                    {
                        _matrix[j, k] -= mult * _matrix[i, k];
                    }
                    y[j] -= mult * y[i];
                }

                // Perform the back substitution. 
                for (int x = _rowCount - 1; x >= 0; x--)
                {
                    float sum = 0f;
                    for (int j = x + 1; j < _rowCount; j++)
                        sum += _matrix[x, j] * w[j];
                    w[x] = (y[x] - sum) / _matrix[x, x];
                }
            }
            
            return true;
        }
    }

    [Serializable]
    public enum TransformAxis
    {
        X,
        Y,
        Z
    }

    [Serializable]
    public enum RBFKernel
    {
        Linear,
        Gaussian,
        ThinPlate,
        MultiquadraticBiharmonic,
        InverseMultiQuadraticBiharmonic,
        BeckertWendlandC2Basis
    }
    
    [Serializable]
    public enum RBFInterpolation
    {
        Linear,
        InverseQuadratic,
        Quadratic,
        SmoothStep,
        SmootherStep
    }

    [Serializable]
    public enum RBFRotationType
    {
        SwingAndTwist,
        Swing,
        Twist
    }
    
    [Serializable]
    public class RBFPose
    {
        public Transform target;
        public RBFRotationType poseMode;
    }
    
    // Drive mesh blendshape weights based on a set of poses.
    public class RBF : MonoBehaviour
    {
        [SerializeField] private SkinnedMeshRenderer mesh;
        [SerializeField] private string[] blendshapeNames;
        [SerializeField] private RBFKernel kernel;
        [SerializeField] private RBFInterpolation interpolation;
        [SerializeField] private float scale = 1f;
        [SerializeField] private float bias;
        [SerializeField] private TransformAxis twistAxis;
        [SerializeField] private bool twistAxisNegate;
        [SerializeField] private Transform[] drivers;
        [SerializeField] private RBFPose[] poses;

        private float[] _driverVector;
        private int _solveCount;
        private int _poseCount;
        private RBFMatrix _poseData;
        private RBFMatrix _poseValues;
        private float[] _weights;
        private float _meanDistance;
        private Vector3 _baseVector;

        private void Awake()
        {
            InitializeRBFData();
        }

        private void Update()
        {
            CalculateDriverVectors();
            SetBlendshapeWeights();
        }

        private void InitializeRBFData()
        {
            GetPoseVectors();
            _solveCount = _poseCount;
            
            // Calculate RBF.
            if (_poseCount == 0) return;

            _weights = new float[_solveCount];
            
            // Create a distance matrix from all poses and
            // calculate the mean distance for the rbf function.
            RBFMatrix distanceMatrix = GetDistances();
            
            // Transform the distance matrix to include the
            // activation values.
            GetActivations(distanceMatrix);
            
            // Create a matrix to store the weights and
            // solve each dimension.
            RBFMatrix wMat = new RBFMatrix(_poseCount, _solveCount);

            for (int i = 0; i < _solveCount; i++)
            {
                // Get the pose values for each dimension.
                float[] y = _poseValues.GetColumnVector(i);

                // Copy the activation matrix because it gets
                // modified during the solving process.
                RBFMatrix solveMat = distanceMatrix.Copy();

                float[] w = new float [_poseCount];
                bool solved = solveMat.Solve(y, ref w);
                if (!solved)
                {
                    Debug.LogError("RBF decomposition failed");
                    return;
                }

                // Store the weights in the weight matrix.
                for (int j = 0; j < _poseCount; j++)
                {
                    wMat.SetElem(j, i, w[i]);
                }
                
            }
        }

        private void SetBlendshapeWeights()
        {
            for (int i = 0; i < _weights.Length; i++)
            {
                float value = _weights[i];

                if (value < 0f)
                {
                    value = 0f;
                }

                if (bias != 0f)
                {
                    value = GetWeightBias(value);
                }
                
                value = InterpolateWeight(value);

                // Set the final weight.
                _weights[i] = value * scale;
            }
        }
        
        private void CalculateDriverVectors()
        {
            Vector3 vec;
            int increment = 0;
            for (int i = 0; i < drivers.Length; i++)
            {
                vec = drivers[i].localRotation * _baseVector;
                _driverVector[0 + increment] = vec.x;
                _driverVector[1 + increment] = vec.y;
                _driverVector[2 + increment] = vec.z;
                _driverVector[3 + increment] = GetTwistAngle(drivers[i].localRotation);
                increment += 4;
            }

        }

        private float GetTwistAngle(Quaternion rot)
        {
            float axisComponent = rot.x;
            if (twistAxis == TransformAxis.Y)
                axisComponent = rot.y;
            else if (twistAxis == TransformAxis.Z)
                axisComponent = rot.z;
            return 2f * Mathf.Atan2(axisComponent, rot.w);
        }
        
        private float GetWeightBias(float value)
        {
            const double epsilon = 2.2204460492503131e-16;
            
            if (bias >= 0f)
            {
                value = Mathf.Abs(value) * Mathf.Pow(Mathf.Abs(value), bias);
            }
            else
            {
                float baseVal = 1 - Mathf.Abs(value);
                // If the value is 1 the bias transformation has zero at the
                // base and outputs NaN.
                if (Mathf.Abs(baseVal) > epsilon)
                    value = 1 - Mathf.Pow(baseVal, (1 + Mathf.Abs(bias)));
                else
                    value = 1;
            }

            return value;
        }

        private float InterpolateWeight(float value)
        {
            switch (interpolation)
            {
                case RBFInterpolation.InverseQuadratic:
                    return 1 - Mathf.Pow((1 - value), 2f);
                case RBFInterpolation.Quadratic:
                    return 1 - Mathf.Pow((1 - value), 1 / 2f);
                case RBFInterpolation.SmoothStep:
                    return value * value * (3 - 2 * value);
                case RBFInterpolation.SmootherStep:
                    return value * value * value * (value * (value * 6 - 15) + 10);
                default:
                    return value;
            }
        }
    }
}