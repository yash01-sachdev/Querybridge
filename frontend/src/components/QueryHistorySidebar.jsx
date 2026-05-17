import { ArrowRight, Clock3, History } from "lucide-react";

import { DatabaseBadge } from "./Badges";

export default function QueryHistorySidebar({ isOpen, items, onClose, onSelect }) {
  return (
    <>
      <button
        type="button"
        aria-label="Close history"
        className={isOpen ? "drawer-backdrop visible" : "drawer-backdrop"}
        onClick={onClose}
      />

      <aside className={isOpen ? "history-drawer open" : "history-drawer"}>
        <div className="history-sidebar-header">
          <h2 className="history-sidebar-title">
            <History className="section-icon" />
            Query History
          </h2>
        </div>

        <div className="history-list">
          {items.map((item) => (
            <button
              key={item.id}
              type="button"
              className="history-card"
              onClick={() => {
                onSelect(item);
                onClose();
              }}
            >
              <div className="history-card-main">
                <div className="history-title" title={item.title}>
                  {item.title}
                </div>

                <div className="history-meta">
                  <span className="history-time">
                    <Clock3 className="tiny-icon" />
                    {item.timestamp}
                  </span>
                  <DatabaseBadge backend={item.backend} />
                </div>
              </div>

              <span className="history-arrow">
                <ArrowRight className="tiny-icon" />
              </span>
            </button>
          ))}
        </div>
      </aside>
    </>
  );
}
