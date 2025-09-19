#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>

// Parámetros del SOM
#define LADO_MAPA 100        // Tamaño del mapa (20x20)
#define NUM_ENTRADAS 18     // Número de características de entrada (columnas "against_")
#define PERIODO 1000        // Número de iteraciones de entrenamiento
#define LEARNING_RATE 0.5   // Tasa de aprendizaje inicial

// Función para calcular la distancia euclidiana en la GPU
__device__ float distancia_euclidiana(float* a, float* b, int n) {
    float sum = 0.0;
    for (int i = 0; i < n; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// Kernel CUDA para calcular la BMU (Best Matching Unit)
__global__ void calcular_bmu_kernel(float* patron, float* matriz_pesos, int* bmu_idx, float* min_dist) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < LADO_MAPA * LADO_MAPA) {
        float dist = distancia_euclidiana(patron, &matriz_pesos[idx * NUM_ENTRADAS], NUM_ENTRADAS);
        float prev_min = atomicMin(reinterpret_cast<int*>(min_dist), __float_as_int(dist));
        if (dist == __int_as_float(prev_min) || dist < __int_as_float(prev_min)) {
            *bmu_idx = idx;
        }
    }
}

// Kernel CUDA para calcular todas las distancias (usado para error topológico)
__global__ void calcular_distancias_kernel(float* patron, float* matriz_pesos, float* distancias) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < LADO_MAPA * LADO_MAPA) {
        distancias[idx] = distancia_euclidiana(patron, &matriz_pesos[idx * NUM_ENTRADAS], NUM_ENTRADAS);
    }
}

// Kernel CUDA para actualizar los pesos de las neuronas
__global__ void actualizar_pesos_kernel(float* patron, float* matriz_pesos, int bmu_x, int bmu_y, float lr, float v) {
    int x = blockIdx.x;
    int y = threadIdx.x;
    int idx = x * LADO_MAPA + y;
    float distancia = sqrt(pow(x - bmu_x, 2) + pow(y - bmu_y, 2));
    if (distancia <= v) {
        float influencia = exp(-distancia * distancia / (2 * v * v));
        for (int i = 0; i < NUM_ENTRADAS; i++) {
            matriz_pesos[idx * NUM_ENTRADAS + i] += lr * influencia * (patron[i] - matriz_pesos[idx * NUM_ENTRADAS + i]);
        }
    }
}

// Función para cargar datos desde un archivo CSV
std::vector<float> cargar_datos(const std::string& archivo) {
    std::vector<float> datos;
    std::ifstream file(archivo);
    if (!file.is_open()) {
        std::cerr << "Error: No se pudo abrir el archivo " << archivo << std::endl;
        return datos;
    }

    std::string linea;
    bool primera_linea = true;
    while (std::getline(file, linea)) {
        if (primera_linea) {  // Omitir el encabezado
            primera_linea = false;
            continue;
        }
        std::stringstream ss(linea);
        std::string valor;
        int columna = 0;
        int inicio_columna = 1;  // Las columnas "against_" empiezan después de "id" (índice 1)
        while (std::getline(ss, valor, ',')) {
            if (columna >= inicio_columna && columna < inicio_columna + NUM_ENTRADAS) {
                try {
                    datos.push_back(std::stof(valor));
                } catch (const std::exception& e) {
                    std::cerr << "Error al convertir valor: " << valor << " en la linea: " << linea << std::endl;
                }
            }
            columna++;
        }
    }
    file.close();
    return datos;
}

// Función para entrenar el SOM y devolver la matriz de pesos entrenada
float* entrenar_som(float* datos, int num_datos) {
    // Reservar memoria en la GPU para la matriz de pesos
    float* d_matriz_pesos;
    cudaMalloc(&d_matriz_pesos, LADO_MAPA * LADO_MAPA * NUM_ENTRADAS * sizeof(float));

    // Inicializar la matriz de pesos con valores aleatorios
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::vector<float> h_matriz_pesos(LADO_MAPA * LADO_MAPA * NUM_ENTRADAS);
    for (int i = 0; i < LADO_MAPA * LADO_MAPA * NUM_ENTRADAS; i++) {
        h_matriz_pesos[i] = dis(gen);
    }
    cudaMemcpy(d_matriz_pesos, h_matriz_pesos.data(), LADO_MAPA * LADO_MAPA * NUM_ENTRADAS * sizeof(float), cudaMemcpyHostToDevice);

    // Reservar memoria en la GPU para la BMU
    int* d_bmu_idx;
    float* d_min_dist;
    cudaMalloc(&d_bmu_idx, sizeof(int));
    cudaMalloc(&d_min_dist, sizeof(float));

    // Copiar datos de entrada a la GPU
    float* d_datos;
    cudaMalloc(&d_datos, num_datos * NUM_ENTRADAS * sizeof(float));
    cudaMemcpy(d_datos, datos, num_datos * NUM_ENTRADAS * sizeof(float), cudaMemcpyHostToDevice);

    // Ciclo de entrenamiento
    for (int i = 0; i < PERIODO; i++) {
        for (int j = 0; j < num_datos; j++) {
            float* patron = &d_datos[j * NUM_ENTRADAS];
            // Inicializar distancia mínima a un valor alto
            float max_float = 1e30;
            cudaMemcpy(d_min_dist, &max_float, sizeof(float), cudaMemcpyHostToDevice);

            // Calcular BMU
            int bloques = (LADO_MAPA * LADO_MAPA + 255) / 256;
            calcular_bmu_kernel<<<bloques, 256>>>(patron, d_matriz_pesos, d_bmu_idx, d_min_dist);
            cudaDeviceSynchronize();

            // Obtener índice de la BMU
            int bmu_idx;
            cudaMemcpy(&bmu_idx, d_bmu_idx, sizeof(int), cudaMemcpyDeviceToHost);
            int bmu_x = bmu_idx / LADO_MAPA;
            int bmu_y = bmu_idx % LADO_MAPA;

            // Calcular parámetros de aprendizaje y vecindario
            float lr_actual = LEARNING_RATE * (1.0 - (float)i / PERIODO);
            float v_actual = (LADO_MAPA / 2.0) * (1.0 - (float)i / PERIODO);

            // Actualizar pesos
            dim3 grid(LADO_MAPA);
            dim3 block(LADO_MAPA);
            actualizar_pesos_kernel<<<grid, block>>>(patron, d_matriz_pesos, bmu_x, bmu_y, lr_actual, v_actual);
            cudaDeviceSynchronize();
        }
        if ((i + 1) % 200 == 0) {
            std::cout << "Iteracion " << i + 1 << "/" << PERIODO << std::endl;
        }
    }

    // Liberar memoria de la GPU (excepto d_matriz_pesos, que se devolverá)
    cudaFree(d_bmu_idx);
    cudaFree(d_min_dist);
    cudaFree(d_datos);

    return d_matriz_pesos;
}

// Función para calcular el error de cuantificación
float calcular_error_cuantificacion(float* datos, int num_datos, float* d_matriz_pesos) {
    float* d_datos;
    cudaMalloc(&d_datos, num_datos * NUM_ENTRADAS * sizeof(float));
    cudaMemcpy(d_datos, datos, num_datos * NUM_ENTRADAS * sizeof(float), cudaMemcpyHostToDevice);

    int* d_bmu_idx;
    float* d_min_dist;
    cudaMalloc(&d_bmu_idx, sizeof(int));
    cudaMalloc(&d_min_dist, sizeof(float));

    float error_total = 0.0;
    for (int j = 0; j < num_datos; j++) {
        float* patron = &d_datos[j * NUM_ENTRADAS];
        float max_float = 1e30;
        cudaMemcpy(d_min_dist, &max_float, sizeof(float), cudaMemcpyHostToDevice);

        int bloques = (LADO_MAPA * LADO_MAPA + 255) / 256;
        calcular_bmu_kernel<<<bloques, 256>>>(patron, d_matriz_pesos, d_bmu_idx, d_min_dist);
        cudaDeviceSynchronize();

        float min_dist;
        cudaMemcpy(&min_dist, d_min_dist, sizeof(float), cudaMemcpyDeviceToHost);
        error_total += min_dist;
    }

    cudaFree(d_datos);
    cudaFree(d_bmu_idx);
    cudaFree(d_min_dist);

    return error_total / num_datos;
}

// Función para calcular el error topológico
float calcular_error_topologico(float* datos, int num_datos, float* d_matriz_pesos) {
    float* d_datos;
    cudaMalloc(&d_datos, num_datos * NUM_ENTRADAS * sizeof(float));
    cudaMemcpy(d_datos, datos, num_datos * NUM_ENTRADAS * sizeof(float), cudaMemcpyHostToDevice);

    float* d_distancias;
    cudaMalloc(&d_distancias, LADO_MAPA * LADO_MAPA * sizeof(float));

    int errores = 0;
    for (int j = 0; j < num_datos; j++) {
        float* patron = &d_datos[j * NUM_ENTRADAS];
        int bloques = (LADO_MAPA * LADO_MAPA + 255) / 256;
        calcular_distancias_kernel<<<bloques, 256>>>(patron, d_matriz_pesos, d_distancias);
        cudaDeviceSynchronize();

        std::vector<float> h_distancias(LADO_MAPA * LADO_MAPA);
        cudaMemcpy(h_distancias.data(), d_distancias, LADO_MAPA * LADO_MAPA * sizeof(float), cudaMemcpyDeviceToHost);

        // Encontrar las dos mejores BMUs
        int bmu1_idx = 0;
        float bmu1_dist = h_distancias[0];
        for (int i = 1; i < LADO_MAPA * LADO_MAPA; i++) {
            if (h_distancias[i] < bmu1_dist) {
                bmu1_dist = h_distancias[i];
                bmu1_idx = i;
            }
        }

        int bmu2_idx = 0;
        float bmu2_dist = std::numeric_limits<float>::max();
        for (int i = 0; i < LADO_MAPA * LADO_MAPA; i++) {
            if (i == bmu1_idx) continue;
            if (h_distancias[i] < bmu2_dist) {
                bmu2_dist = h_distancias[i];
                bmu2_idx = i;
            }
        }

        // Verificar si las BMUs son adyacentes
        int bmu1_x = bmu1_idx / LADO_MAPA;
        int bmu1_y = bmu1_idx % LADO_MAPA;
        int bmu2_x = bmu2_idx / LADO_MAPA;
        int bmu2_y = bmu2_idx % LADO_MAPA;
        int dx = abs(bmu1_x - bmu2_x);
        int dy = abs(bmu1_y - bmu2_y);
        if (dx > 1 || dy > 1) {
            errores++;
        }
    }

    cudaFree(d_datos);
    cudaFree(d_distancias);

    return (float)errores / num_datos;
}

int main() {
    // Cargar datos desde un archivo CSV
    std::vector<float> datos_vec = cargar_datos("pokemon_train.csv");
    if (datos_vec.empty()) {
        std::cerr << "No se cargaron datos. Terminando el programa." << std::endl;
        return 1;
    }
    int num_datos = datos_vec.size() / NUM_ENTRADAS;
    float* datos = datos_vec.data();

    // Entrenar el SOM
    float* d_matriz_pesos = entrenar_som(datos, num_datos);

    // Calcular métricas
    float error_cuant = calcular_error_cuantificacion(datos, num_datos, d_matriz_pesos);
    float error_topo = calcular_error_topologico(datos, num_datos, d_matriz_pesos);

    // Mostrar métricas
    std::cout << "Error de Cuantificacion: " << error_cuant << std::endl;
    std::cout << "Error Topologico: " << error_topo << std::endl;

    // Liberar memoria de la matriz de pesos
    cudaFree(d_matriz_pesos);

    std::cout << "Entrenamiento completado." << std::endl;
    return 0;
}