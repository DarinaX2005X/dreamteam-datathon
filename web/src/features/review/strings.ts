export const roleLabels = {
  cashier: "Кассир",
  client: "Клиент",
  collector: "Инкассатор",
  controller: "Контролёр",
  staff: "Другой сотрудник",
  outsider: "Посторонний",
  unknown: "Не определено",
} as const;
export const phaseLabels: Record<string, string> = {
  queued: "Ожидает запуска",
  decode: "Читаем запись",
  perception: "Находим людей и предметы",
  facts: "Восстанавливаем действия",
  rules: "Сверяем с нормами",
  evidence: "Готовим кадры",
  done: "Анализ завершён",
  failed: "Анализ прерван",
  cancelled: "Анализ отменён",
};
