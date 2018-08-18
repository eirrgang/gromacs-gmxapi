#include "gmxapi/system.h"

#include <array>

#include "md-impl.h"
#include "system-impl.h"
#include "workflow.h"

#include "gmxapi/context.h"
#include "gmxapi/md.h"
#include "gmxapi/session.h"
#include "gmxapi/status.h"

#include "gromacs/compat/make_unique.h"
#include "gromacs/utility.h"
#include "gromacs/mdrun/runner.h"

namespace gmxapi
{

System::Impl::~Impl() = default;


// Constructor and destructor needs to be defined after Impl is defined so that we can
// use unique_ptr
System::System() :
    impl_ {gmx::compat::make_unique<System::Impl>()}
{
    assert(impl_ != nullptr);
}

std::shared_ptr<Session> System::launch(std::shared_ptr<Context> context)
{
    return impl_->launch(std::move(context));
}

std::shared_ptr<Session> System::launch()
{
    return impl_->launch();
}

Status System::status()
{
    assert(impl_ != nullptr);
    return impl_->status();
}

System::System(std::unique_ptr<System::Impl> &&implementation) :
    impl_ {std::forward < std::unique_ptr < System::Impl>>(implementation)}
{
    assert(impl_ != nullptr);
}

Status System::setRestraint(std::shared_ptr<gmxapi::MDModule> module)
{
    return impl_->setRestraint(std::move(module));
}

std::shared_ptr<MDWorkSpec> System::getSpec()
{
    return impl_->getSpec();
}

System::~System() = default;

System::System(System &&) noexcept = default;

System &System::operator=(System &&) noexcept = default;

std::unique_ptr<gmxapi::System> fromTprFile(std::string filename)
{
    // Confirm the file is readable and parseable and note unique identifying information
    // for when the work spec is used in a different environment.

    // Create a new Workflow instance.
    // \todo: error handling
    auto workflow = Workflow::create(filename);

    // This may produce errors or throw exceptions in the future, but in 0.0.3 only memory allocation
    // errors are possible, and we do not have a plan for how to recover from them.
    auto systemImpl = gmx::compat::make_unique<System::Impl>(std::move(workflow));
    assert(systemImpl != nullptr);
    auto system = gmx::compat::make_unique<System>(std::move(systemImpl));

    // The TPR file has enough information for us to
    //  1. choose an MD engine
    //  2. Get structure information
    //  3. Get topology information
    //  4. Get a lot of simulation and runtime parameters, but not all.
    // It does not have enough information on its own to determine much about the
    // necessary computation environment. That comes from environment
    // introspection and user runtime options.

    return system;
}


System::Impl::Impl() :
    context_ {std::make_shared<Context>()},
workflow_ {
    nullptr
},
status_ {
    gmx::compat::make_unique<Status>()
}
{
    assert(context_ != nullptr);
    assert(status_ != nullptr);
}

Status System::Impl::status() const
{
    return *status_;
}

System::Impl::Impl(std::unique_ptr<gmxapi::Workflow> &&workflow) noexcept :
    context_ {defaultContext()},
workflow_ {
    std::move(workflow)
},
spec_ {
    std::make_shared<MDWorkSpec>()
},
status_ {
    gmx::compat::make_unique<Status>(true)
}
{
    assert(context_ != nullptr);
    assert(workflow_ != nullptr);
    assert(spec_ != nullptr);
    assert(status_ != nullptr);
}

std::shared_ptr<Session> System::Impl::launch(std::shared_ptr<Context> context)
{
    std::shared_ptr<Session> session {
        nullptr
    };
    if (context != nullptr)
    {
        session = context->launch(*workflow_);
        assert(session);

        for (auto && module : spec_->getModules())
        {
            setSessionRestraint(session.get(), module);
        }
    }
    else
    {
        // we should log the error and return nullptr, but we have nowhere to set
        // a status object, by the described behavior. Should both native context and
        // provided context receive error status?
    }

    return session;
}

std::shared_ptr<Session> System::Impl::launch()
{
    assert(context_ != nullptr);
    auto session = launch(context_);
    return session;
}

Status System::Impl::setRestraint(std::shared_ptr<gmxapi::MDModule> module)
{
    assert(spec_ != nullptr);
    spec_->addModule(std::move(module));
    return Status(true);
}

std::shared_ptr<MDWorkSpec> System::Impl::getSpec()
{
    return spec_;
}

} // end namespace gmxapi