import axios, { type AxiosRequestConfig } from "axios";

export const apiClient = axios.create({
  baseURL: "/api",
  headers: {
    "Content-Type": "application/json"
  }
});

export const api = {
  post: <T>(url: string, data?: unknown, config?: AxiosRequestConfig) =>
    apiClient.post<T>(url, data, config).then((res) => res.data),
  get: <T>(url: string, params?: Record<string, unknown>, config?: AxiosRequestConfig) =>
    apiClient.get<T>(url, { ...config, params }).then((res) => res.data)
};
