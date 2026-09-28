export function LoadingState({ text }: { text: string }) {
  return (
    <div className="loading-card card">
      <span className="spinner" />
      <span>{text}</span>
    </div>
  );
}
